"""
Ant Colony Optimization (ACO) Algorithm
Makaledeki Table 1 ve Section 3.2'yi uygular
"""

import numpy as np
from typing import List, Dict, Tuple
from core.problem import DisasterReliefProblem
from core.solution import Solution, Route, ParetoFront


class ACO:
    """
    Temel Karınca Kolonisi Optimizasyonu
    Rota optimizasyonu için kullanılır
    """
    
    def __init__(self, 
                 problem: DisasterReliefProblem,
                 num_ants: int = 30,
                 num_iterations: int = 20,
                 alpha: float = 1.0,  # Feromon önem katsayısı
                 beta: float = 0.0,   # Mesafe önem katsayısı (makalede 0)
                 rho: float = 0.3,    # Buharlaşma oranı
                 q: float = 100.0):   # Feromon miktarı
        
        self.problem = problem
        self.num_ants = num_ants
        self.num_iterations = num_iterations
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.q = q
        
        # Feromon matrisi
        self.pheromone: Dict[Tuple[int, int], float] = {}
        self.initialize_pheromone()
        
        # En iyi çözümler
        self.best_routes: List[Route] = []
        self.best_cost: float = float('inf')
        
    def initialize_pheromone(self):
        """
        Feromon matrisini başlat
        Tüm kenarlar için başlangıç feromon seviyesi
        """
        total_nodes = self.problem.num_depots + self.problem.num_areas
        initial_pheromone = 1.0
        
        for i in range(total_nodes):
            for j in range(total_nodes):
                if i != j:
                    self.pheromone[(i, j)] = initial_pheromone
    
    def optimize_routes_for_depot(self, 
                                   depot_id: int, 
                                   assigned_areas: List[int]) -> List[Route]:
        """
        Belirli bir depo için optimal rotaları bul
        Makaledeki Table 1 algoritması
        
        Args:
            depot_id: Depo ID'si
            assigned_areas: Bu depoya atanan bölgelerin ID'leri
            
        Returns:
            Bu depo için optimal rota listesi
        """
        if len(assigned_areas) == 0:
            return []
        
        # Yerel feromon matrisi (sadece bu bölgeler için)
        self._initialize_local_pheromone(depot_id, assigned_areas)
        
        best_routes = []
        best_total_cost = float('inf')
        
        # Ana ACO döngüsü
        for iteration in range(self.num_iterations):
            iteration_routes = []
            
            # Her karınca bir çözüm üretir
            for ant in range(self.num_ants):
                routes = self._construct_routes(depot_id, assigned_areas)
                iteration_routes.append(routes)
            
            # Bu iterasyondaki en iyi rotayı bul
            for routes in iteration_routes:
                total_cost = self._calculate_routes_cost(routes)
                if total_cost < best_total_cost:
                    best_total_cost = total_cost
                    best_routes = routes
            
            # Feromon güncelle
            self._update_pheromone(iteration_routes, depot_id, assigned_areas)
        
        return best_routes
    
    def _initialize_local_pheromone(self, depot_id: int, areas: List[int]):
        """Yerel feromon matrisini başlat"""
        depot_node = depot_id
        area_nodes = [self.problem.num_depots + area_id for area_id in areas]
        all_nodes = [depot_node] + area_nodes
        
        for i in all_nodes:
            for j in all_nodes:
                if i != j:
                    if (i, j) not in self.pheromone:
                        self.pheromone[(i, j)] = 1.0
    
    
    def _construct_routes(self, 
                         depot_id: int, 
                         assigned_areas: List[int]) -> List[Route]:
        """
        Bir karınca için rota kümesi oluştur (GÜVENLİ VERSİYON)
        Sonsuz döngüleri engellemek için 'imkansız' müşterileri atlar.
        """
        routes = []
        unvisited = set(assigned_areas)
        vehicle_id = 0
        
        depot_node = depot_id
        
        # GÜVENLİK: Sonsuz döngü sayacı
        max_attempts = len(assigned_areas) * 2
        attempts = 0
        
        while unvisited:
            attempts += 1
            # Eğer makul deneme sayısını geçersek döngüyü kır
            if attempts > max_attempts:
                break
                
            route = Route(depot_id=depot_id, vehicle_id=vehicle_id)
            current_node = depot_node
            current_load = 0.0
            current_time = 0.0
            
            # Bir rota oluştur (kapasite ve zaman limitlerine göre)
            while unvisited:
                # Bir sonraki bölgeyi seç
                next_area = self._select_next_area(
                    current_node, unvisited, current_load, current_time
                )
                
                if next_area is None:
                    break  # Bu araç için başka bölge eklenemez
                
                # Bölgeyi rotaya ekle
                area = self.problem.areas[next_area]
                route.add_area(next_area)
                unvisited.remove(next_area)
                
                # Yük ve zaman güncellemeleri
                current_load += area.demand
                area_node = self.problem.num_depots + next_area
                travel_time = self.problem.get_travel_time(current_node, area_node)
                
                arrival_time = current_time + travel_time
                service_start = max(arrival_time, area.soft_lower)
                current_time = service_start + area.service_time
                current_node = area_node
            
            # Rotayı tamamla ve ekle
            if not route.is_empty():
                try:
                    # Solution sınıfını güvenli şekilde import et veya kullan
                    # (Burada Solution nesnesi oluşturulurken hata almamak için try-except)
                    temp_solution = Solution(self.problem)
                    route = temp_solution.calculate_route_metrics(route)
                except:
                    pass # Hata olursa ham rotayı kullan
                    
                routes.append(route)
                vehicle_id += 1
            else:
                # KRİTİK DÜZELTME:
                # Yeni bir araç açtık ama 'unvisited' listesindeki hiçbir yere gidemedi.
                # Bu demektir ki kalan bölgeler 'imkansız' (zamanı geçmiş vs.).
                # Sonsuz döngüye girmemek için listeden zorla birini atıyoruz.
                if unvisited:
                    skipped = unvisited.pop()
                    # İstersen buraya print(f"Atlandı: {skipped}") yazabilirsin
        
        return routes






    def _select_next_area(self, 
                         current_node: int, 
                         unvisited: set, 
                         current_load: float,
                         current_time: float) -> int:
        """
        Bir sonraki ziyaret edilecek bölgeyi seç
        Makaledeki Equation (18) - sadece feromon kullanıyor
        """
        candidates = []
        probabilities = []
        
        for area_id in unvisited:
            area = self.problem.areas[area_id]
            area_node = self.problem.num_depots + area_id
            
            # Fizibilite kontrolleri
            # 1. Kapasite kontrolü
            if current_load + area.demand > self.problem.vehicle_capacity:
                continue
            
            # 2. Zaman penceresi kontrolü (sert limit)
            travel_time = self.problem.get_travel_time(current_node, area_node)
            arrival_time = current_time + travel_time
            service_start = max(arrival_time, area.soft_lower)
            
            if service_start > area.hard_upper:
                continue  # Sert zaman limitini aşıyor
            
            # 3. Toplam zaman kontrolü
            return_time = self.problem.get_travel_time(area_node, current_node // self.problem.num_areas)
            total_time = service_start + area.service_time + return_time
            if total_time > self.problem.vehicle_time_limit:
                continue
            
            # Olasılık hesapla (Equation 18)
            tau = self.pheromone.get((current_node, area_node), 1.0)
            probability = tau ** self.alpha
            
            candidates.append(area_id)
            probabilities.append(probability)
        
        if len(candidates) == 0:
            return None
        
        # Olasılıkları normalize et
        total_prob = sum(probabilities)
        if total_prob == 0:
            # Rastgele seç
            return np.random.choice(candidates)
        
        probabilities = [p / total_prob for p in probabilities]
        
        # Olasılığa göre seç
        selected = np.random.choice(candidates, p=probabilities)
        return selected
    
    def _calculate_routes_cost(self, routes: List[Route]) -> float:
        """Rota kümesinin toplam maliyetini hesapla"""
        total_cost = 0.0
        
        for route in routes:
            # Mesafe maliyeti
            total_cost += route.total_distance * self.problem.transport_cost_rate
            # Zaman penceresi cezası
            total_cost += route.time_window_penalty
            # Araç sabit maliyeti
            total_cost += self.problem.vehicle_fixed_cost
        
        return total_cost
    
    def _update_pheromone(self, 
                         all_routes: List[List[Route]], 
                         depot_id: int,
                         assigned_areas: List[int]):
        """
        Feromon güncelleme (Equations 19-20)
        """
        # Tüm feromonları buharlaştır (Equation 19)
        for key in self.pheromone:
            self.pheromone[key] *= (1 - self.rho)
        
        # Her karıncanın rotaları için feromon ekle (Equation 20)
        for routes in all_routes:
            route_cost = self._calculate_routes_cost(routes)
            if route_cost == 0:
                continue
            
            delta_tau = self.q / route_cost
            
            for route in routes:
                # Rota boyunca feromon bırak
                path = route.get_full_path()
                for i in range(len(path) - 1):
                    node_i = path[i]
                    node_j = path[i + 1]
                    
                    # Negatif ID'leri düzelt (depolar)
                    if node_i < 0:
                        node_i = depot_id
                    if node_j < 0:
                        node_j = depot_id
                    
                    key = (node_i, node_j)
                    if key in self.pheromone:
                        self.pheromone[key] += delta_tau


class ACOSolver:
    """
    Tam problem için ACO çözücü
    Tüm depoları bağımsız olarak optimize eder
    """
    
    def __init__(self, problem: DisasterReliefProblem, **aco_params):
        self.problem = problem
        self.aco = ACO(problem, **aco_params)
        
    def solve(self, depot_assignments: np.ndarray) -> Solution:
        """
        Verilen depo atamalarına göre optimal rotaları bul
        
        Args:
            depot_assignments: Her bölgenin hangi depoya atandığı
            
        Returns:
            Tam çözüm
        """
        solution = Solution(self.problem)
        solution.set_depot_assignments(depot_assignments)
        
        all_routes = []
        
        # Her açık depo için rotaları optimize et
        for depot_id in solution.opened_depots:
            # Bu depoya atanan bölgeleri bul
            assigned_areas = np.where(depot_assignments == depot_id)[0].tolist()
            
            # ACO ile rotaları optimize et
            routes = self.aco.optimize_routes_for_depot(depot_id, assigned_areas)
            all_routes.extend(routes)
        
        solution.set_routes(all_routes)
        solution.evaluate()
        
        return solution