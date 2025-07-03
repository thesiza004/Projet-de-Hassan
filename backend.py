
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import pandas as pd
import numpy as np
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import requests
import pickle
import concurrent.futures
import os
import tempfile
import polyline
import json
import time
from datetime import datetime
from itertools import combinations

try:
    import polyline
    POLYLINE_AVAILABLE = True
except ImportError:
    POLYLINE_AVAILABLE = False
    print("Warning: polyline library not available. Install with: pip install polyline")

app = Flask(__name__)
CORS(app)

class VRPSolverAPI:
    def __init__(self):
        self.osrm_server_url = "http://router.project-osrm.org/route/v1/driving/{lon1},{lat1};{lon2},{lat2}?overview=full"
        self.num_vehicles = 3
        self.vehicle_capacities = [2000, 2500, 2000]
        self.vehicle_ids = [1, 2, 3]
        self.depot_index = 0
        self.depot_time_window_start = "08:00"
        self.depot_time_window_end = "18:00"
        self.distance_cache = {}
        self.speed = 50.0
        self.cache_file = 'distance_cache.pkl'
        self.time_limit = 60
        self.service_time = 10
        self.geometry_cache = {}
        self.df = None
        self.routes = []
        self.solution_data = None
        self.vehicles_df = None
        self.load_cache()

    def load_cache(self):
        """Charge le cache depuis un fichier avec validation."""
        try:
            with open(self.cache_file, 'rb') as f:
                self.distance_cache = pickle.load(f)
            invalid_keys = [k for k, v in self.distance_cache.items() if not isinstance(v, dict) or v.get('distance', 0) <= 0]
            for k in invalid_keys:
                del self.distance_cache[k]
            print(f"Cache chargé: {len(self.distance_cache)} entrées valides, {len(invalid_keys)} supprimées")
            self.save_cache()
        except FileNotFoundError:
            print("Aucun cache trouvé, démarrage avec un cache vide.")
            self.distance_cache = {}
        except Exception as e:
            print(f"Erreur lors du chargement du cache: {str(e)}")
            self.distance_cache = {}

    def save_cache(self):
        """Sauvegarde le cache dans un fichier."""
        try:
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.distance_cache, f)
            print(f"Cache sauvegardé: {len(self.distance_cache)} entrées")
        except Exception as e:
            print(f"Erreur lors de la sauvegarde du cache: {str(e)}")

    def valid_coordinates(self, lat1, lon1, lat2, lon2):
        """Valide que les coordonnées sont dans des plages raisonnables"""
        for lat, lon in [(lat1, lon1), (lat2, lon2)]:
            if not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
                return False
        return True

    def osrm_distance(self, lat1, lon1, lat2, lon2):
        """Appel OSRM pour une paire de points avec retries et pause ajustée"""
        if not self.valid_coordinates(lat1, lon1, lat2, lon2):
            raise ValueError("Coordonnées invalides")

        osrm_url = f"http://router.project-osrm.org/route/v1/driving/{lon1},{lat1};{lon2},{lat2}?overview=full"

        for attempt in range(2):
            try:
                response = requests.get(osrm_url, timeout=7)
                response.raise_for_status()
                print(f"🛰️ Requête OSRM (tentative {attempt+1}): {osrm_url}")
                data = response.json()
                if data.get("code") != "Ok" or len(data["routes"]) == 0:
                    print(f"⛔ Réponse OSRM invalide: {data}")
                    raise ValueError("Pas d'itinéraire trouvé via OSRM")

                return {
                    'distance': data["routes"][0]["distance"],
                    'geometry': data["routes"][0]["geometry"]
                }
            except Exception as e:
                print(f"Échec OSRM (tentative {attempt+1}): {str(e)}")
                if attempt == 1:
                    print(f"Échec final pour ({lat1}, {lon1}) -> ({lat2}, {lon2})")
                    return {'distance': 9999, 'geometry': None}
                time.sleep(5)

    def osrm_distance_with_cache(self, lat1, lon1, lat2, lon2):
        """Retourne un dictionnaire avec 'distance' et 'geometry' depuis le cache ou OSRM"""
        key = (lat1, lon1, lat2, lon2)
        reverse_key = (lat2, lon2, lat1, lon1)

        if key in self.distance_cache:
            return self.distance_cache[key]
        if reverse_key in self.distance_cache:
            return self.distance_cache[reverse_key]

        try:
            result = self.osrm_distance(lat1, lon1, lat2, lon2)
            self.distance_cache[key] = result
            return result
        except Exception as e:
            print(f"Erreur OSRM entre {key}: {str(e)}")
            return {'distance': 9999, 'geometry': None}

    def create_distance_matrix(self, locations):
        """Crée une matrice de distance avec OSRM Table par lots de 10 points ou fallback vers requêtes individuelles"""
        num_locations = len(locations)
        matrix = np.zeros((num_locations, num_locations))
        self.geometry_cache = {}
        
        self.load_cache()
        
        print(f"Calcul OSRM pour {num_locations}x{num_locations} distances...")
        
        def fill_submatrix(start_idx, end_idx):
            sub_locations = locations[start_idx:end_idx]
            coords = ";".join(f"{lon},{lat}" for lat, lon in sub_locations)
            osrm_url = f"http://router.project-osrm.org/table/v1/driving/{coords}?annotations=distance"
            
            for attempt in range(2):
                try:
                    response = requests.get(osrm_url, timeout=30)
                    response.raise_for_status()
                    data = response.json()
                    if data.get("code") != "Ok":
                        raise ValueError(f"Réponse OSRM Table invalide: {data}")
                    print(f"✅ Table réussie pour points {start_idx} à {end_idx-1}")
                    return np.array(data["distances"])
                except Exception as e:
                    print(f"Échec OSRM Table (tentative {attempt+1}, points {start_idx}-{end_idx-1}): {str(e)}")
                    if attempt == 1:
                        return None
                    time.sleep(5)
        
        batch_size = 15
        for i in range(0, num_locations, batch_size):
            for j in range(i, num_locations, batch_size):
                start_i, end_i = i, min(i + batch_size, num_locations)
                start_j, end_j = j, min(j + batch_size, num_locations)
                
                if i == j:
                    sub_matrix = fill_submatrix(start_i, end_i)
                    if sub_matrix is not None:
                        matrix[start_i:end_i, start_i:end_i] = sub_matrix
                    continue
                
                sub_locations = locations[start_i:end_i] + locations[start_j:end_j]
                sub_matrix = fill_submatrix(0, len(sub_locations))
                if sub_matrix is not None:
                    matrix[start_i:end_i, start_j:end_j] = sub_matrix[:end_i-start_i, end_i-start_i:]
                    matrix[start_j:end_j, start_i:end_i] = sub_matrix[end_i-start_i:, :end_i-start_i]
        
        if np.any((matrix == 0) & (~np.eye(num_locations, dtype=bool))):
            print("⚠️ Certaines distances manquantes, utilisation des requêtes individuelles")
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                futures = {}
                for i in range(num_locations):
                    for j in range(i + 1, num_locations):
                        if matrix[i][j] == 0:
                            lat1, lon1 = locations[i]
                            lat2, lon2 = locations[j]
                            futures[(i,j)] = executor.submit(self.osrm_distance_with_cache, lat1, lon1, lat2, lon2)
                
                for (i,j), future in futures.items():
                    try:
                        result = future.result()
                        if not isinstance(result, dict) or 'distance' not in result:
                            raise ValueError("Format de résultat OSRM invalide")
                        
                        matrix[i][j] = matrix[j][i] = result['distance']
                        self.geometry_cache[(i,j)] = result.get('geometry', None)
                        
                    except Exception as e:
                        print(f"Erreur entre {i} et {j}: {str(e)}")
                        matrix[i][j] = matrix[j][i] = 9999
                        self.geometry_cache[(i,j)] = None
                    time.sleep(0.1)
        
        if np.any((matrix == 0) & (~np.eye(num_locations, dtype=bool))):
            print("⚠️ WARNING: La matrice contient des zéros pour des segments non diagonaux")
        
        print("✅ Matrice de distances OSRM :")
        print(matrix)
        self.save_cache()
        
        return matrix.astype(int).tolist()

    def create_time_matrix(self, distance_matrix):
        """Convertit la matrice de distance en matrice de temps (minutes)"""
        speed_m_per_min = (self.speed * 1000) / 60
        time_matrix = [[int(distance / speed_m_per_min) if distance > 0 else 0 for distance in row] for row in distance_matrix]
        return time_matrix

    def time_to_minutes(self, time_str):
        """Convertit une chaîne HH:MM en minutes depuis minuit"""
        import re
        if not isinstance(time_str, str) or not re.match(r'^\d{1,2}:\d{2}$', time_str):
            return 8 * 60
        try:
            h, m = map(int, time_str.split(':'))
            return h * 60 + m
        except:
            return 8 * 60

    def minutes_to_time_str(self, minutes):
        """Convertit des minutes en chaîne HH:MM"""
        hours, mins = divmod(int(minutes), 60)
        return f"{hours:02d}:{mins:02d}"

    def get_time_windows(self):
        """Récupère les fenêtres temporelles"""
        time_windows = []
        depot_open = self.time_to_minutes(self.depot_time_window_start)
        depot_close = self.time_to_minutes(self.depot_time_window_end)
        
        for i in range(len(self.df)):
            if i == self.depot_index:
                time_windows.append((depot_open, depot_close))
            else:
                start = self.time_to_minutes(str(self.df.iloc[i].get('Time_Window_Start', self.depot_time_window_start)))
                end = self.time_to_minutes(str(self.df.iloc[i].get('Time_Window_End', self.depot_time_window_end)))
                time_windows.append((start, end))
        
        return time_windows

    def solve_vrp(self, distance_matrix, time_matrix, demands, time_windows, num_vehicles, vehicle_capacities, depot_index, service_time):
        """Résout le problème VRP avec OR-Tools"""
        manager = pywrapcp.RoutingIndexManager(len(distance_matrix), num_vehicles, depot_index)
        routing = pywrapcp.RoutingModel(manager)
        
        def distance_callback(from_index, to_index):
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return distance_matrix[from_node][to_node]
        
        transit_callback_index = routing.RegisterTransitCallback(distance_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
        
        def time_callback(from_index, to_index):
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            travel_time = time_matrix[from_node][to_node]
            if to_node != depot_index:
                return travel_time + service_time
            return travel_time
        
        time_callback_index = routing.RegisterTransitCallback(time_callback)
        
        def demand_callback(from_index):
            from_node = manager.IndexToNode(from_index)
            return demands[from_node]
        
        demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
        routing.AddDimensionWithVehicleCapacity(
            demand_callback_index,
            0,
            vehicle_capacities,
            True,
            'Capacity'
        )
        
        routing.AddDimension(
            time_callback_index,
            30,
            1440,
            False,
            'Time'
        )
        time_dimension = routing.GetDimensionOrDie('Time')
        
        for location_idx, time_window in enumerate(time_windows):
            if location_idx < len(time_windows):
                index = manager.NodeToIndex(location_idx)
                if index >= 0:
                    start_time, end_time = time_window
                    time_dimension.CumulVar(index).SetRange(start_time, end_time)
        
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        )
        search_parameters.local_search_metaheuristic = (
            routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
        )
        search_parameters.time_limit.seconds = self.time_limit
        
        solution = routing.SolveWithParameters(search_parameters)
        
        return {
            'manager': manager,
            'routing': routing,
            'solution': solution,
            'num_vehicles': num_vehicles,
            'depot_index': depot_index,
            'service_time': service_time,
            'distance_matrix': distance_matrix,
            'time_matrix': time_matrix
        }

    def process_solution(self, solution_data, locations, demands, names):
        """Traite la solution pour l'affichage et la visualisation"""
        manager = solution_data['manager']
        routing = solution_data['routing']
        solution = solution_data['solution']
        
        if not solution:
            return {"error": "Aucune solution trouvée!"}
        
        total_distance = 0
        total_load = 0
        self.routes = []
        
        time_dimension = routing.GetDimensionOrDie('Time')
        capacity_dimension = routing.GetDimensionOrDie('Capacity')
        
        vehicle_stats = []
        routes_data = []
        
        for vehicle_id in range(solution_data['num_vehicles']):
            index = routing.Start(vehicle_id)
            if solution.Value(routing.NextVar(index)) == routing.End(vehicle_id):
                vehicle_stats.append({
                    'id': self.vehicle_ids[vehicle_id],
                    'used': False,
                    'capacity': self.vehicle_capacities[vehicle_id],
                    'load': 0,
                    'fill_rate': 0.0,
                    'distance': 0,
                    'clients': 0
                })
                continue
            
            route_distance = 0
            route_load = 0
            route_points = []
            client_count = 0
            driver_id = self.vehicle_ids[vehicle_id]
            
            while not routing.IsEnd(index):
                node_index = manager.IndexToNode(index)
                next_node_index = manager.IndexToNode(solution.Value(routing.NextVar(index)))
                
                time_var = time_dimension.CumulVar(index)
                load_var = capacity_dimension.CumulVar(index)
                
                arrival_time = solution.Min(time_var)
                load = solution.Value(load_var)
                time_str = self.minutes_to_time_str(arrival_time)
                
                route_points.append({
                    'index': node_index,
                    'name': names[node_index],
                    'location': locations[node_index],
                    'arrival': time_str,
                    'load': load,
                    'driver_id': driver_id
                })
                
                if node_index != solution_data['depot_index']:
                    route_load += demands[node_index]
                    client_count += 1
                
                route_distance += solution_data['distance_matrix'][node_index][next_node_index]
                index = solution.Value(routing.NextVar(index))
            
            node_index = manager.IndexToNode(index)
            time_var = time_dimension.CumulVar(index)
            arrival_time = solution.Min(time_var)
            time_str = self.minutes_to_time_str(arrival_time)
            
            route_points.append({
                'index': node_index,
                'name': names[node_index],
                'location': locations[node_index],
                'arrival': time_str,
                'load': solution.Value(capacity_dimension.CumulVar(index)),
                'driver_id': driver_id
            })
            
            fill_rate = (route_load / self.vehicle_capacities[vehicle_id]) * 100 if self.vehicle_capacities[vehicle_id] > 0 else 0
            
            total_distance += route_distance
            total_load += route_load
            self.routes.append(route_points)
            
            routes_data.append({
                'vehicle_id': vehicle_id + 1,
                'driver_id': driver_id,
                'points': route_points,
                'distance': route_distance,
                'load': route_load,
                'fill_rate': fill_rate,
                'clients': client_count
            })
            
            vehicle_stats.append({
                'id': driver_id,
                'used': True,
                'capacity': self.vehicle_capacities[vehicle_id],
                'load': route_load,
                'fill_rate': fill_rate,
                'distance': route_distance,
                'clients': client_count
            })
        
        vehicles_used = len(self.routes)
        total_clients = sum(len([p for p in route if p['index'] != self.depot_index]) for route in self.routes)
        avg_fill_rate = sum(stat['fill_rate'] for stat in vehicle_stats if stat['used']) / max(vehicles_used, 1)
        total_capacity = sum(self.vehicle_capacities[:vehicles_used])
        global_efficiency = (total_load / total_capacity * 100) if total_capacity > 0 else 0
        
        self.solution_data = {
            'routes': self.routes,
            'total_distance': total_distance,
            'total_load': total_load,
            'locations': locations,
            'names': names,
            'depot_index': solution_data['depot_index'],
            'vehicle_stats': vehicle_stats,
            'global_efficiency': global_efficiency
        }
        
        return {
            'success': True,
            'routes': routes_data,
            'summary': {
                'vehicles_used': vehicles_used,
                'total_vehicles': solution_data['num_vehicles'],
                'total_distance': total_distance,
                'total_load': total_load,
                'total_clients': total_clients,
                'avg_fill_rate': avg_fill_rate,
                'global_efficiency': global_efficiency,
                'depot_name': names[self.depot_index]
            },
            'vehicle_stats': vehicle_stats
        }

vrp_solver = VRPSolverAPI()
@app.route('/accueil')
def accueil():
    return send_file('index.html')
@app.route('/api/upload-points', methods=['POST'])
def upload_points():
    """Upload et traitement du fichier des points de livraison"""
    tmp_file_path = None
    try:
        file = request.files['file']
        import uuid
        temp_filename = f"points_{uuid.uuid4().hex}.xlsx"
        tmp_file_path = os.path.join(tempfile.gettempdir(), temp_filename)
        
        file.save(tmp_file_path)
        vrp_solver.df = pd.read_excel(tmp_file_path)
        
        required_columns = ['Latitude', 'Longitude']
        if not all(col in vrp_solver.df.columns for col in required_columns):
            return jsonify({'error': 'Le fichier doit contenir les colonnes: Latitude et Longitude'}), 400
        
        for i, row in vrp_solver.df.iterrows():
            lat, lon = row['Latitude'], row['Longitude']
            if not (-90 <= lat <= 90 and -180 <= lon <= 180):
                return jsonify({'error': f'Coordonnées invalides pour le point {i}: ({lat}, {lon})'}), 400
        
        locations = vrp_solver.df[['Latitude', 'Longitude']].values
        for i in range(len(locations)):
            for j in range(i + 1, len(locations)):
                if np.allclose(locations[i], locations[j], atol=1e-4):
                    return jsonify({'error': f'Points {i} et {j} sont trop proches: {locations[i]}'}), 400
        
        vrp_solver.depot_index = 0
        
        if 'Demande' not in vrp_solver.df.columns:
            vrp_solver.df['Demande'] = 1
            vrp_solver.df.loc[0, 'Demande'] = 0
        else:
            vrp_solver.df.loc[0, 'Demande'] = 0
        
        if 'Time_Window_Start' not in vrp_solver.df.columns:
            vrp_solver.df['Time_Window_Start'] = vrp_solver.depot_time_window_start
        if 'Time_Window_End' not in vrp_solver.df.columns:
            vrp_solver.df['Time_Window_End'] = vrp_solver.depot_time_window_end
        if 'Nom' not in vrp_solver.df.columns:
            vrp_solver.df['Nom'] = [f"Point {i}" for i in range(len(vrp_solver.df))]
            vrp_solver.df.loc[0, 'Nom'] = "DÉPÔT"
        else:
            if not str(vrp_solver.df.loc[0, 'Nom']).upper().startswith('DÉPÔT'):
                vrp_solver.df.loc[0, 'Nom'] = f"DÉPÔT - {vrp_solver.df.loc[0, 'Nom']}"
        
        return jsonify({
            'success': True,
            'message': f'Données chargées: {len(vrp_solver.df)} points',
            'depot_info': {
                'name': vrp_solver.df.loc[0, 'Nom'],
                'latitude': float(vrp_solver.df.loc[0, 'Latitude']),
                'longitude': float(vrp_solver.df.loc[0, 'Longitude'])
            },
            'points_count': len(vrp_solver.df)
        })
        
    except Exception as e:
        return jsonify({'error': f'Erreur lors du chargement: {str(e)}'}), 500
    finally:
        if tmp_file_path and os.path.exists(tmp_file_path):
            try:
                time.sleep(0.1)
                os.unlink(tmp_file_path)
            except OSError:
                pass

@app.route('/api/upload-vehicles', methods=['POST'])
def upload_vehicles():
    """Upload et traitement du fichier des véhicules"""
    tmp_file_path = None
    try:
        file = request.files['file']
        import uuid
        temp_filename = f"vehicles_{uuid.uuid4().hex}.xlsx"
        tmp_file_path = os.path.join(tempfile.gettempdir(), temp_filename)
        
        file.save(tmp_file_path)
        vrp_solver.vehicles_df = pd.read_excel(tmp_file_path)
        
        required_columns = ['id_driver', 'capacity']
        if not all(col in vrp_solver.vehicles_df.columns for col in required_columns):
            return jsonify({'error': 'Le fichier doit contenir les colonnes: id_driver et capacity'}), 400
        
        if vrp_solver.vehicles_df['capacity'].isnull().any() or (vrp_solver.vehicles_df['capacity'] <= 0).any():
            return jsonify({'error': 'Toutes les capacités doivent être des nombres positifs'}), 400
        
        vrp_solver.vehicle_ids = vrp_solver.vehicles_df['id_driver'].tolist()
        vrp_solver.vehicle_capacities = vrp_solver.vehicles_df['capacity'].tolist()
        vrp_solver.num_vehicles = len(vrp_solver.vehicles_df)
        
        return jsonify({
            'success': True,
            'message': f'Véhicules importés: {len(vrp_solver.vehicles_df)}',
            'vehicles_count': len(vrp_solver.vehicles_df),
            'total_capacity': sum(vrp_solver.vehicle_capacities),
            'avg_capacity': np.mean(vrp_solver.vehicle_capacities)
        })
        
    except Exception as e:
        return jsonify({'error': f'Erreur lors du chargement des véhicules: {str(e)}'}), 500
    finally:
        if tmp_file_path and os.path.exists(tmp_file_path):
            try:
                time.sleep(0.1)
                os.unlink(tmp_file_path)
            except OSError:
                pass

@app.route('/api/configure', methods=['POST'])
def configure_solver():
    """Configuration des paramètres du solveur"""
    try:
        config = request.json
        if 'num_vehicles' in config:
            vrp_solver.num_vehicles = config['num_vehicles']
        if 'vehicle_capacities' in config:
            vrp_solver.vehicle_capacities = config['vehicle_capacities']
        if 'depot_time_window_start' in config:
            vrp_solver.depot_time_window_start = config['depot_time_window_start']
        if 'depot_time_window_end' in config:
            vrp_solver.depot_time_window_end = config['depot_time_window_end']
        if 'speed' in config:
            vrp_solver.speed = config['speed']
        if 'time_limit' in config:
            vrp_solver.time_limit = config['time_limit']
        if 'service_time' in config:
            vrp_solver.service_time = config['service_time']
        
        if len(vrp_solver.vehicle_ids) != vrp_solver.num_vehicles:
            vrp_solver.vehicle_ids = list(range(1, vrp_solver.num_vehicles + 1))
        
        return jsonify({'success': True, 'message': 'Configuration mise à jour'})
        
    except Exception as e:
        return jsonify({'error': f'Erreur de configuration: {str(e)}'}), 500

@app.route('/api/solve', methods=['POST'])
def solve():
    """Lance la résolution du VRP"""
    try:
        if vrp_solver.df is None:
            return jsonify({'error': 'Aucun fichier de points chargé'}), 400
        
        locations = vrp_solver.df[['Latitude', 'Longitude']].values.tolist()
        demands = vrp_solver.df['Demande'].values.tolist()
        names = vrp_solver.df['Nom'].values.tolist()
        
        if len(locations) < 2:
            return jsonify({'error': 'Au moins 2 points sont nécessaires'}), 400
        
        print("Création de la matrice de distances...")
        distance_matrix = vrp_solver.create_distance_matrix(locations)
        
        distance_array = np.array(distance_matrix)
        zero_non_diagonal = np.sum((distance_array == 0) & (~np.eye(len(distance_array), dtype=bool)))
        if zero_non_diagonal > 0:
            return jsonify({'error': f'Matrice de distances invalide: {zero_non_diagonal} segments ont une distance nulle'}), 400
        
        print("Création de la matrice de temps...")
        time_matrix = vrp_solver.create_time_matrix(distance_matrix)
        time_windows = vrp_solver.get_time_windows()
        
        print("Lancement de l'optimisation OR-Tools...")
        solution = vrp_solver.solve_vrp(
            distance_matrix,
            time_matrix,
            demands,
            time_windows,
            vrp_solver.num_vehicles,
            vrp_solver.vehicle_capacities,
            vrp_solver.depot_index,
            vrp_solver.service_time
        )
        
        print("Traitement de la solution...")
        result = vrp_solver.process_solution(solution, locations, demands, names)
        
        return jsonify(result)
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Erreur détaillée: {error_details}")
        return jsonify({'error': f'Erreur pendant la résolution: {str(e)}'}), 500

@app.route('/api/route-geometries', methods=['GET'])
def get_route_geometries():
    """Retourne les géométries des routes finales pour l'affichage sur la carte"""
    try:
        if not vrp_solver.solution_data or not vrp_solver.routes:
            return jsonify({'error': 'Aucune solution ou routes disponibles'}), 400
        
        routes_geometries = []
        
        segments_needed = set()
        for route in vrp_solver.routes:
            for i in range(len(route) - 1):
                from_idx = route[i]['index']
                to_idx = route[i + 1]['index']
                segments_needed.add(tuple(sorted([from_idx, to_idx])))
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            futures = {}
            for from_idx, to_idx in segments_needed:
                geometry_key = (from_idx, to_idx)
                reverse_key = (to_idx, from_idx)
                if geometry_key not in vrp_solver.geometry_cache and reverse_key not in vrp_solver.geometry_cache:
                    lat1, lon1 = vrp_solver.solution_data['locations'][from_idx]
                    lat2, lon2 = vrp_solver.solution_data['locations'][to_idx]
                    futures[geometry_key] = executor.submit(vrp_solver.osrm_distance, lat1, lon1, lat2, lon2)
            
            for geometry_key, future in futures.items():
                try:
                    result = future.result()
                    vrp_solver.geometry_cache[geometry_key] = result.get('geometry', None)
                except Exception as e:
                    print(f"Erreur géométrie {geometry_key}: {str(e)}")
                    vrp_solver.geometry_cache[geometry_key] = None
        
        for vehicle_id, route in enumerate(vrp_solver.routes):
            vehicle_geometries = []
            for i in range(len(route) - 1):
                from_idx = route[i]['index']
                to_idx = route[i + 1]['index']
                
                geometry_key = (from_idx, to_idx)
                reverse_key = (to_idx, from_idx)
                
                geometry = vrp_solver.geometry_cache.get(geometry_key, None)
                if geometry is None:
                    geometry = vrp_solver.geometry_cache.get(reverse_key, None)
                
                coordinates = []
                if geometry and POLYLINE_AVAILABLE:
                    try:
                        coordinates = polyline.decode(geometry)
                        coordinates = [[lon, lat] for lat, lon in coordinates]
                    except:
                        coordinates = [
                            [vrp_solver.solution_data['locations'][from_idx][1], vrp_solver.solution_data['locations'][from_idx][0]],
                            [vrp_solver.solution_data['locations'][to_idx][1], vrp_solver.solution_data['locations'][to_idx][0]]
                        ]
                else:
                    coordinates = [
                        [vrp_solver.solution_data['locations'][from_idx][1], vrp_solver.solution_data['locations'][from_idx][0]],
                        [vrp_solver.solution_data['locations'][to_idx][1], vrp_solver.solution_data['locations'][to_idx][0]]
                    ]
                
                vehicle_geometries.append({
                    'from': from_idx,
                    'to': to_idx,
                    'coordinates': coordinates,
                    'from_name': route[i]['name'],
                    'to_name': route[i + 1]['name']
                })
            
            routes_geometries.append({
                'vehicle_id': vehicle_id + 1,
                'driver_id': route[0]['driver_id'] if route else None,
                'segments': vehicle_geometries
            })
        
        vrp_solver.save_cache()
        return jsonify({
            'success': True,
            'routes_geometries': routes_geometries
        })
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Erreur géométries: {error_details}")
        return jsonify({'error': f'Erreur lors de la récupération des géométries: {str(e)}'}), 500

@app.route('/api/export', methods=['GET'])
def export_solution():
    """Exporte la solution au format Excel"""
    tmp_file_path = None
    try:
        if not vrp_solver.routes:
            return jsonify({'error': 'Aucune solution disponible'}), 400
        
        import uuid
        temp_filename = f"export_{uuid.uuid4().hex}.xlsx"
        tmp_file_path = os.path.join(tempfile.gettempdir(), temp_filename)
        
        data = []
        for vehicle_id, route in enumerate(vrp_solver.routes):
            for stop_id, point in enumerate(route):
                data.append({
                    'ID_Conducteur': point['driver_id'],
                    'Véhicule': vehicle_id + 1,
                    'Ordre': stop_id + 1,
                    'Nom': point['name'],
                    'Latitude': point['location'][0],
                    'Longitude': point['location'][1],
                    'Heure_Arrivée': point['arrival'],
                    'Charge_Cumulée': point['load'],
                    'Type': 'Dépôt' if point['index'] == vrp_solver.solution_data['depot_index'] else 'Client'
                })
        
        solution_df = pd.DataFrame(data)
        
        with pd.ExcelWriter(tmp_file_path, engine='openpyxl') as writer:
            solution_df.to_excel(writer, sheet_name='Solution_Détaillée', index=False)
            
            summary_data = []
            for vehicle_id, route in enumerate(vrp_solver.routes):
                if vehicle_id < len(vrp_solver.solution_data['vehicle_stats']):
                    vehicle_stat = vrp_solver.solution_data['vehicle_stats'][vehicle_id]
                    clients = [point for point in route if point['index'] != vrp_solver.solution_data['depot_index']]
                    duration = vrp_solver.time_to_minutes(route[-1]['arrival']) - vrp_solver.time_to_minutes(route[0]['arrival']) if len(route) >= 2 else 0
                    
                    summary_data.append({
                        'ID_Conducteur': vehicle_stat['id'],
                        'Véhicule': vehicle_id + 1,
                        'Statut': 'Utilisé' if vehicle_stat['used'] else 'Non utilisé',
                        'Capacité_Max': vehicle_stat['capacity'],
                        'Charge_Transportée': vehicle_stat['load'],
                        'Taux_Remplissage_%': round(vehicle_stat['fill_rate'], 1),
                        'Nombre_Clients': len(clients),
                        'Distance_km': round(vehicle_stat['distance'] / 1000, 2),
                        'Heure_Départ': route[0]['arrival'] if route else '',
                        'Heure_Retour': route[-1]['arrival'] if route else '',
                        'Durée_min': duration
                    })
            
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='Analyse_Véhicules', index=False)
        
        return send_file(tmp_file_path, 
                        as_attachment=True, 
                        download_name=f'vrp_solution_{datetime.now().strftime("%Y%m%d_%H%M%S")}.xlsx',
                        mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Erreur export: {error_details}")
        return jsonify({'error': f'Erreur lors de l\'export: {str(e)}'}), 500
    finally:
        if tmp_file_path:
            import threading
            def delayed_cleanup():
                time.sleep(10)
                try:
                    if os.path.exists(tmp_file_path):
                        os.unlink(tmp_file_path)
                except OSError:
                    pass
            cleanup_thread = threading.Thread(target=delayed_cleanup)
            cleanup_thread.daemon = True
            cleanup_thread.start()

@app.route('/api/status', methods=['GET'])
def get_status():
    """Retourne le statut actuel du solveur"""
    try:
        status = {
            'solver_ready': True,
            'points_loaded': vrp_solver.df is not None,
            'vehicles_loaded': vrp_solver.vehicles_df is not None,
            'solution_available': vrp_solver.solution_data is not None,
            'cache_entries': len(vrp_solver.distance_cache),
            'config': {
                'num_vehicles': vrp_solver.num_vehicles,
                'vehicle_capacities': vrp_solver.vehicle_capacities,
                'vehicle_ids': vrp_solver.vehicle_ids,
                'speed': vrp_solver.speed,
                'time_limit': vrp_solver.time_limit,
                'service_time': vrp_solver.service_time
            }
        }
        
        if vrp_solver.df is not None:
            status['points_count'] = len(vrp_solver.df)
            status['depot_info'] = {
                'name': vrp_solver.df.loc[0, 'Nom'],
                'latitude': float(vrp_solver.df.loc[0, 'Latitude']),
                'longitude': float(vrp_solver.df.loc[0, 'Longitude'])
            }
        
        return jsonify(status)
        
    except Exception as e:
        return jsonify({'error': f'Erreur status: {str(e)}'}), 500

if __name__ == '__main__':
    print("🚚 Démarrage du serveur VRP API...")
    print("📡 Endpoints disponibles:")
    print("  POST /api/upload-points - Upload fichier points")
    print("  POST /api/upload-vehicles - Upload fichier véhicules")
    print("  POST /api/configure - Configuration solveur")
    print("  POST /api/solve - Lancer optimisation")
    print("  GET  /api/route-geometries - Récupérer géométries")
    print("  GET  /api/export - Exporter solution")
    print("  GET  /api/status - Statut du solveur")
    print("🌟 Serveur prêt sur http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)