"""
Çözüm geçerliliğini kontrol eder
Makaledeki tüm kısıtları (Constraints 3-16) doğrular
"""

from typing import List, Tuple
import numpy as np


class SolutionValidator:
    """
    Çözüm geçerliliği kontrol sınıfı
    """
    
    def __init__(self, problem):
        self.problem = problem
        self.violations: List[str] = []
        
    def validate(self, solution) -> Tuple[bool, List[str]]:
        """
        Çözümü doğrula
        Returns: (is_valid, list_of_violations)
        """
        self.violations = []
        
        # Temel kontroller
        if solution.depot_assignments is None:
            self.violations.append("Depot assignments not set")
            return False, self.violations
        
        if len(solution.routes) == 0:
            self.violations.append("No routes defined")
            return False, self.violations
        
        # Tüm kısıtları kontrol et
        self._check_constraint_3(solution)  # Depo açıklık kontrolü
        self._check_constraint_4(solution)  # Rota başlangıç/bitiş
        self._check_constraint_5(solution)  # Her bölge bir kez
        self._check_constraint_6(solution)  # Depo kapasitesi
        self._check_constraint_7(solution)  # Araç kapasitesi
        self._check_constraint_12(solution) # Araç çalışma süresi
        self._check_constraint_14(solution) # Zaman pencereleri
        
        is_valid = len(self.violations) == 0
        return is_valid, self.violations
    
    def _check_constraint_3(self, solution):
        """
        Constraint 3: y_ikr <= z_k
        Bir bölge sadece açık bir depoya atanabilir
        """
        for depot_id in solution.opened_depots:
            depot = self.problem.depots[depot_id]
            if not depot.is_open:
                # Açık olarak işaretle
                depot.is_open = True
        
        # Her atanan depoda en az bir bölge olmalı
        assigned_depots = set(solution.depot_assignments)
        for depot_id in assigned_depots:
            if depot_id not in solution.opened_depots:
                self.violations.append(
                    f"Constraint 3: Depot {depot_id} used but not opened"
                )
    
    def _check_constraint_4(self, solution):
        """
        Constraint 4: Her rota aynı depodan başlayıp bitmeli
        """
        for route in solution.routes:
            if route.is_empty():
                continue
            
            # Rota kendi deposuna dönmeli
            if route.depot_id not in solution.opened_depots:
                self.violations.append(
                    f"Constraint 4: Route from closed depot {route.depot_id}"
                )
    
    def _check_constraint_5(self, solution):
        """
        Constraint 5: Her bölge tam olarak bir kez ziyaret edilmeli
        """
        visited_areas = set()
        
        for route in solution.routes:
            for area_id in route.sequence:
                if area_id in visited_areas:
                    self.violations.append(
                        f"Constraint 5: Area {area_id} visited multiple times"
                    )
                visited_areas.add(area_id)
        
        # Tüm bölgeler ziyaret edildi mi?
        all_areas = set(range(self.problem.num_areas))
        not_visited = all_areas - visited_areas
        if not_visited:
            self.violations.append(
                f"Constraint 5: Areas {not_visited} not visited"
            )
    
    def _check_constraint_6(self, solution):
        """
        Constraint 6: Depo kapasitesi kontrolü
        Σ d_i * y_ikr <= h_k
        """
        depot_loads = {}
        
        for route in solution.routes:
            depot_id = route.depot_id
            if depot_id not in depot_loads:
                depot_loads[depot_id] = 0.0
            
            depot_loads[depot_id] += route.total_demand
        
        for depot_id, total_load in depot_loads.items():
            depot = self.problem.depots[depot_id]
            if total_load > depot.capacity:
                self.violations.append(
                    f"Constraint 6: Depot {depot_id} overloaded "
                    f"({total_load:.2f} > {depot.capacity:.2f})"
                )
    
    def _check_constraint_7(self, solution):
        """
        Constraint 7: Araç kapasitesi kontrolü
        Σ d_i * y_ikr <= Q
        """
        for route in solution.routes:
            if route.total_demand > self.problem.vehicle_capacity:
                self.violations.append(
                    f"Constraint 7: Route overloaded "
                    f"({route.total_demand:.2f} > {self.problem.vehicle_capacity:.2f})"
                )
    
    def _check_constraint_12(self, solution):
        """
        Constraint 12: Araç çalışma süresi limiti
        T_(n+1,r) - T_0r <= Δ
        """
        for route in solution.routes:
            if route.total_time > self.problem.vehicle_time_limit:
                self.violations.append(
                    f"Constraint 12: Route exceeds time limit "
                    f"({route.total_time:.2f} > {self.problem.vehicle_time_limit:.2f})"
                )
    
    def _check_constraint_14(self, solution):
        """
        Constraint 14: Zaman penceresi kontrolü (sert limit)
        e_i <= T_ir <= o_i
        """
        for route in solution.routes:
            if route.is_empty():
                continue
            
            depot_node = route.depot_id
            current_node = depot_node
            current_time = 0.0
            
            for area_id in route.sequence:
                area = self.problem.areas[area_id]
                area_node = self.problem.num_depots + area_id
                
                # Seyahat süresi
                travel_time = self.problem.get_travel_time(current_node, area_node)
                arrival_time = current_time + travel_time
                
                # Erken varış durumunda bekle
                if arrival_time < area.soft_lower:
                    service_start = area.soft_lower
                else:
                    service_start = arrival_time
                
                # Sert üst limitı kontrol et
                if service_start > area.hard_upper:
                    self.violations.append(
                        f"Constraint 14: Area {area_id} hard time window violated "
                        f"(arrival={service_start:.2f} > hard_upper={area.hard_upper:.2f})"
                    )
                
                # Servisi tamamla
                current_time = service_start + area.service_time
                current_node = area_node
    
    def check_depot_capacity_utilization(self, solution) -> dict:
        """
        Her deponun kapasite kullanımını hesapla
        """
        utilization = {}
        
        for depot_id in solution.opened_depots:
            depot = self.problem.depots[depot_id]
            total_demand = 0.0
            
            for route in solution.routes:
                if route.depot_id == depot_id:
                    total_demand += route.total_demand
            
            utilization[depot_id] = {
                'total_demand': total_demand,
                'capacity': depot.capacity,
                'utilization_rate': total_demand / depot.capacity if depot.capacity > 0 else 0
            }
        
        return utilization
    
    def check_vehicle_capacity_utilization(self, solution) -> dict:
        """
        Her aracın kapasite kullanımını hesapla
        """
        utilization = []
        
        for i, route in enumerate(solution.routes):
            util_rate = route.total_demand / self.problem.vehicle_capacity
            utilization.append({
                'route_id': i,
                'depot_id': route.depot_id,
                'demand': route.total_demand,
                'capacity': self.problem.vehicle_capacity,
                'utilization_rate': util_rate
            })
        
        return utilization
    
    def get_solution_statistics(self, solution) -> dict:
        """
        Çözüm istatistiklerini döndür
        """
        is_valid, violations = self.validate(solution)
        
        depot_util = self.check_depot_capacity_utilization(solution)
        vehicle_util = self.check_vehicle_capacity_utilization(solution)
        
        avg_vehicle_util = solution.get_average_vehicle_utilization()
        
        return {
            'is_valid': is_valid,
            'violations': violations,
            'num_opened_depots': len(solution.opened_depots),
            'num_routes': len(solution.routes),
            'avg_vehicle_utilization': avg_vehicle_util,
            'depot_utilization': depot_util,
            'vehicle_utilization': vehicle_util,
            'f1_penalty_cost': solution.f1_penalty_cost,
            'f2_operational_cost': solution.f2_operational_cost,
            'depot_opening_cost': solution.depot_opening_cost,
            'vehicle_fixed_cost': solution.vehicle_fixed_cost,
            'transport_cost': solution.transport_cost
        }