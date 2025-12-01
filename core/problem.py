"""
Problem tanımları ve veri yapıları
Makaledeki model formülasyonunu temsil eder
"""

import numpy as np
from typing import List, Tuple, Dict
from dataclasses import dataclass, field


@dataclass
class Area:
    """Etkilenen bölge (affected area)"""
    id: int
    location: Tuple[float, float]  # (x, y) koordinatları
    demand: float  # d_i - talep
    service_time: float  # s_i - servis süresi
    
    # Zaman pencereleri
    soft_lower: float  # e_i - yumuşak alt sınır
    soft_upper: float  # l_i - yumuşak üst sınır  
    hard_upper: float  # o_i - sert üst sınır
    
    def __post_init__(self):
        """Zaman penceresi doğrulaması"""
        if not (self.soft_lower <= self.soft_upper <= self.hard_upper):
            raise ValueError(f"Area {self.id}: Invalid time windows")
    
    def get_time_window_penalty(self, arrival_time: float, penalty_rate: float) -> float:
        """
        Zaman penceresi cezasını hesapla (Equation 15)
        """
        if arrival_time < self.soft_lower:
            return 0.0  # Erken varış, bekleme var ama ceza yok
        elif arrival_time <= self.soft_upper:
            return 0.0  # Zamanında varış
        elif arrival_time <= self.hard_upper:
            return penalty_rate * (arrival_time - self.soft_upper)
        else:
            return float('inf')  # Kabul edilemez


@dataclass
class Depot:
    """Aday depo (candidate depot)"""
    id: int
    location: Tuple[float, float]  # (x, y) koordinatları
    capacity: float  # h_k - depolama kapasitesi
    opening_cost: float  # g_k - açılış maliyeti
    is_open: bool = False  # z_k - açık mı?


@dataclass  
class Vehicle:
    """Homojen araç (homogeneous vehicle)"""
    id: int
    capacity: float  # Q - yük kapasitesi
    fixed_cost: float  # φ - sabit maliyet
    working_time_limit: float  # Δ - çalışma süresi limiti
    depot_id: int = None  # Hangi depodan çıkıyor
    

class DisasterReliefProblem:
    """
    Ana problem sınıfı
    Makaledeki LRP modelini temsil eder
    """
    
    def __init__(self, 
                 num_areas: int, 
                 num_depots: int,
                 vehicle_capacity: float = 200,
                 vehicle_fixed_cost: float = 2.0,
                 vehicle_time_limit: float = 480,  # 8 saat = 480 dakika
                 penalty_rate: float = 0.1,
                 transport_cost_rate: float = 0.01):
        
        self.num_areas = num_areas
        self.num_depots = num_depots
        
        # Veriler
        self.areas: List[Area] = []
        self.depots: List[Depot] = []
        
        # Araç parametreleri
        self.vehicle_capacity = vehicle_capacity
        self.vehicle_fixed_cost = vehicle_fixed_cost
        self.vehicle_time_limit = vehicle_time_limit
        
        # Maliyet parametreleri
        self.penalty_rate = penalty_rate  # c_p
        self.transport_cost_rate = transport_cost_rate  # birim mesafe maliyeti
        
        # Mesafe ve zaman matrisleri
        self.distance_matrix: np.ndarray = None
        self.time_matrix: np.ndarray = None
        
    def add_area(self, area: Area):
        """Etkilenen bölge ekle"""
        self.areas.append(area)
        
    def add_depot(self, depot: Depot):
        """Aday depo ekle"""
        self.depots.append(depot)
        
    def compute_distance_matrix(self):
        """
        Tüm lokasyonlar arası mesafe matrisini hesapla
        İlk num_depots satır/sütun depolar, geri kalanı bölgeler
        """
        total_nodes = self.num_depots + self.num_areas
        self.distance_matrix = np.zeros((total_nodes, total_nodes))
        
        # Tüm lokasyonları birleştir: depolar + bölgeler
        all_locations = []
        for depot in self.depots:
            all_locations.append(depot.location)
        for area in self.areas:
            all_locations.append(area.location)
            
        # Euclidean mesafe hesapla
        for i in range(total_nodes):
            for j in range(total_nodes):
                if i != j:
                    loc_i = all_locations[i]
                    loc_j = all_locations[j]
                    distance = np.sqrt((loc_i[0] - loc_j[0])**2 + 
                                     (loc_i[1] - loc_j[1])**2)
                    self.distance_matrix[i][j] = distance
        
        # Zaman matrisi = mesafe matrisi (birim hız varsayımı)
        self.time_matrix = self.distance_matrix.copy()
        
    def get_distance(self, from_node: int, to_node: int) -> float:
        """İki nokta arası mesafe"""
        return self.distance_matrix[from_node][to_node]
    
    def get_travel_time(self, from_node: int, to_node: int) -> float:
        """İki nokta arası seyahat süresi"""
        return self.time_matrix[from_node][to_node]
    
    def get_travel_cost(self, from_node: int, to_node: int) -> float:
        """İki nokta arası seyahat maliyeti (c_ij)"""
        return self.distance_matrix[from_node][to_node] * self.transport_cost_rate
    
    @staticmethod
    def generate_random_instance(num_areas: int, 
                                 num_depots: int,
                                 map_size: float = 100,
                                 seed: int = None) -> 'DisasterReliefProblem':
        """
        Rastgele problem örneği oluştur
        Makaledeki Table 2'ye göre
        """
        if seed is not None:
            np.random.seed(seed)
        
        problem = DisasterReliefProblem(num_areas, num_depots)
        
        # Depoları oluştur
        for i in range(num_depots):
            location = (np.random.rand() * map_size, 
                       np.random.rand() * map_size)
            capacity = 200  # Table 2'den
            opening_cost = 10.0  # Makaleden
            depot = Depot(i, location, capacity, opening_cost)
            problem.add_depot(depot)
        
        # Bölgeleri oluştur  
        for i in range(num_areas):
            location = (np.random.rand() * map_size,
                       np.random.rand() * map_size)
            demand = np.random.uniform(10, 30)  # Table 2'den
            service_time = np.random.uniform(5, 15)
            
            # Zaman pencerelerini oluştur (makaledeki gibi)
            soft_lower = np.random.uniform(20, 300)
            soft_upper = soft_lower + np.random.uniform(30, 60)
            hard_upper = soft_upper + np.random.uniform(10, 30)
            
            area = Area(i, location, demand, service_time,
                       soft_lower, soft_upper, hard_upper)
            problem.add_area(area)
        
        # Mesafe matrisini hesapla
        problem.compute_distance_matrix()
        
        return problem
    
    def __str__(self):
        return (f"DisasterReliefProblem(areas={self.num_areas}, "
                f"depots={self.num_depots})")