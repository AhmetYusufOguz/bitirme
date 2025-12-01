"""
Çözüm gösterimi ve değerlendirme
Makaledeki x, y, T değişkenlerini temsil eder
"""

import numpy as np
from typing import List, Dict, Tuple
from dataclasses import dataclass, field
import copy


@dataclass
class Route:
    """
    Bir aracın rotası
    Makaledeki R_k değişkeni
    """
    depot_id: int
    vehicle_id: int
    sequence: List[int] = field(default_factory=list)  # Bölge ID'leri
    
    # Hesaplanan değerler
    total_demand: float = 0.0
    total_distance: float = 0.0
    total_time: float = 0.0
    time_window_penalty: float = 0.0
    
    def add_area(self, area_id: int):
        """Rotaya bölge ekle"""
        self.sequence.append(area_id)
        
    def is_empty(self) -> bool:
        """Rota boş mu?"""
        return len(self.sequence) == 0
    
    def get_full_path(self) -> List[int]:
        """
        Tam rotayı döndür: depot -> areas -> depot
        Depot ID'si negatif olarak kodlanır
        """
        depot_node = -(self.depot_id + 1)  # -1, -2, -3, ...
        return [depot_node] + self.sequence + [depot_node]


class Solution:
    """
    Problem çözümü
    Makaledeki tüm karar değişkenlerini içerir
    """
    
    def __init__(self, problem):
        self.problem = problem
        
        # Karar değişkenleri
        self.depot_assignments: np.ndarray = None  # Her bölge hangi depoya atandı
        self.opened_depots: List[int] = []  # Açık depoların ID'leri
        self.routes: List[Route] = []  # Tüm araç rotaları
        
        # Amaç fonksiyonu değerleri
        self.f1_penalty_cost: float = float('inf')  # Zaman penceresi cezası
        self.f2_operational_cost: float = float('inf')  # Operasyonel maliyet
        
        # Alt maliyet bileşenleri
        self.depot_opening_cost: float = 0.0
        self.vehicle_fixed_cost: float = 0.0
        self.transport_cost: float = 0.0
        
        # Geçerlilik
        self.is_feasible_solution: bool = False
        
    def initialize_random(self):
        """
        Rastgele başlangıç çözümü oluştur
        Her bölgeyi rastgele bir depoya ata
        """
        num_areas = self.problem.num_areas
        num_depots = self.problem.num_depots
        
        # Her bölge için rastgele depo seç
        self.depot_assignments = np.random.randint(0, num_depots, num_areas)
        
        # Açık depoları belirle
        self.opened_depots = list(np.unique(self.depot_assignments))
        
    def set_depot_assignments(self, assignments: np.ndarray):
        """
        Depo atamalarını ayarla
        assignments[i] = bölge i'nin atandığı depo ID'si
        """
        self.depot_assignments = assignments.copy()
        self.opened_depots = list(np.unique(assignments))
        
    def set_routes(self, routes: List[Route]):
        """Rotaları ayarla"""
        self.routes = routes
        
    def evaluate(self) -> Tuple[float, float]:
        """
        Çözümü değerlendir ve amaç fonksiyonlarını hesapla
        Returns: (f1_penalty_cost, f2_operational_cost)
        """
        if self.depot_assignments is None or len(self.routes) == 0:
            return (float('inf'), float('inf'))
        
        # f1: Zaman penceresi cezası (Objective 1)
        self.f1_penalty_cost = 0.0
        for route in self.routes:
            self.f1_penalty_cost += route.time_window_penalty
            
        # f2: Operasyonel maliyet (Objective 2)
        # f2 = depot_cost + vehicle_cost + transport_cost
        
        # Depo açılış maliyeti
        self.depot_opening_cost = 0.0
        for depot_id in self.opened_depots:
            depot = self.problem.depots[depot_id]
            self.depot_opening_cost += depot.opening_cost
        
        # Araç sabit maliyeti
        self.vehicle_fixed_cost = len(self.routes) * self.problem.vehicle_fixed_cost
        
        # Taşıma maliyeti
        self.transport_cost = 0.0
        for route in self.routes:
            self.transport_cost += route.total_distance * self.problem.transport_cost_rate
        
        self.f2_operational_cost = (self.depot_opening_cost + 
                                   self.vehicle_fixed_cost + 
                                   self.transport_cost)
        
        return (self.f1_penalty_cost, self.f2_operational_cost)
    
    def dominates(self, other: 'Solution') -> bool:
        """
        Bu çözüm diğerini domine ediyor mu?
        Makaledeki Pareto dominans tanımı
        """
        better_in_one = (self.f1_penalty_cost < other.f1_penalty_cost or 
                        self.f2_operational_cost < other.f2_operational_cost)
        
        not_worse_in_both = (self.f1_penalty_cost <= other.f1_penalty_cost and 
                            self.f2_operational_cost <= other.f2_operational_cost)
        
        return not_worse_in_both and better_in_one
    
    def calculate_route_metrics(self, route: Route) -> Route:
        """
        Bir rotanın metriklerini hesapla
        Makaledeki Constraints (12)-(15)'i uygular
        """
        if route.is_empty():
            return route
        
        problem = self.problem
        depot_id = route.depot_id
        depot_node_id = depot_id  # Depolar 0'dan başlar
        
        # Başlangıç değerleri
        route.total_demand = 0.0
        route.total_distance = 0.0
        route.total_time = 0.0
        route.time_window_penalty = 0.0
        
        current_node = depot_node_id
        current_time = 0.0
        
        # Rotadaki her bölgeyi ziyaret et
        for area_id in route.sequence:
            area = problem.areas[area_id]
            area_node_id = problem.num_depots + area_id  # Bölgeler depoların ardından
            
            # Talep ekle
            route.total_demand += area.demand
            
            # Seyahat süresi ve mesafe
            travel_time = problem.get_travel_time(current_node, area_node_id)
            travel_dist = problem.get_distance(current_node, area_node_id)
            
            route.total_distance += travel_dist
            route.total_time += travel_time
            
            # Varış zamanı
            arrival_time = current_time + travel_time
            
            # Zaman penceresi kontrolü (Constraint 14)
            if arrival_time < area.soft_lower:
                # Erken varış - bekle
                service_start_time = area.soft_lower
            else:
                service_start_time = arrival_time
            
            # Zaman penceresi cezası hesapla
            penalty = area.get_time_window_penalty(service_start_time, problem.penalty_rate)
            route.time_window_penalty += penalty
            
            # Servis süresini ekle
            current_time = service_start_time + area.service_time
            current_node = area_node_id
        
        # Depoya dön (Constraint 10)
        return_travel_time = problem.get_travel_time(current_node, depot_node_id)
        return_distance = problem.get_distance(current_node, depot_node_id)
        
        route.total_distance += return_distance
        route.total_time += return_travel_time
        current_time += return_travel_time
        
        return route
    
    def copy(self) -> 'Solution':
        """Çözümün derin kopyasını oluştur"""
        new_sol = Solution(self.problem)
        new_sol.depot_assignments = self.depot_assignments.copy() if self.depot_assignments is not None else None
        new_sol.opened_depots = self.opened_depots.copy()
        new_sol.routes = copy.deepcopy(self.routes)
        new_sol.f1_penalty_cost = self.f1_penalty_cost
        new_sol.f2_operational_cost = self.f2_operational_cost
        new_sol.depot_opening_cost = self.depot_opening_cost
        new_sol.vehicle_fixed_cost = self.vehicle_fixed_cost
        new_sol.transport_cost = self.transport_cost
        new_sol.is_feasible_solution = self.is_feasible_solution
        return new_sol
    
    def get_objectives(self) -> Tuple[float, float]:
        """Amaç fonksiyon değerlerini döndür"""
        return (self.f1_penalty_cost, self.f2_operational_cost)
    
    def get_average_vehicle_utilization(self) -> float:
        """
        Ortalama araç kullanım oranını hesapla
        Başarı kriteri: >= %85
        """
        if len(self.routes) == 0:
            return 0.0
        
        total_utilization = 0.0
        for route in self.routes:
            utilization = route.total_demand / self.problem.vehicle_capacity
            total_utilization += utilization
        
        return total_utilization / len(self.routes)
    
    def __str__(self):
        return (f"Solution(f1={self.f1_penalty_cost:.2f}, "
                f"f2={self.f2_operational_cost:.2f}, "
                f"depots={len(self.opened_depots)}, "
                f"routes={len(self.routes)})")
    
    def __repr__(self):
        return self.__str__()


class ParetoFront:
    """
    Pareto optimal çözümler kümesi
    """
    
    def __init__(self):
        self.solutions: List[Solution] = []
        
    def add(self, solution: Solution):
        """
        Çözüm ekle ve domine edilen çözümleri kaldır
        """
        # Bu çözüm mevcut çözümler tarafından domine ediliyor mu?
        is_dominated = False
        for existing in self.solutions:
            if existing.dominates(solution):
                is_dominated = True
                break
        
        if is_dominated:
            return  # Ekleme
        
        # Bu çözüm tarafından domine edilen çözümleri kaldır
        self.solutions = [s for s in self.solutions if not solution.dominates(s)]
        
        # Yeni çözümü ekle
        self.solutions.append(solution)
    
    def get_solutions(self) -> List[Solution]:
        """Tüm Pareto optimal çözümleri döndür"""
        return self.solutions
    
    def size(self) -> int:
        """Pareto setindeki çözüm sayısı"""
        return len(self.solutions)
    
    def get_objectives_array(self) -> np.ndarray:
        """
        Amaç fonksiyon değerlerini numpy array olarak döndür
        Shape: (n_solutions, 2)
        """
        if len(self.solutions) == 0:
            return np.array([])
        
        objectives = []
        for sol in self.solutions:
            objectives.append([sol.f1_penalty_cost, sol.f2_operational_cost])
        
        return np.array(objectives)