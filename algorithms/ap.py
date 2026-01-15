"""
Alternative Hybrid (AP) Algorithm
ACO depo ataması yapar, PSO rotaları optimize eder
PA-LRP'nin tersi - karşılaştırma için
"""

import numpy as np
from typing import List
from core.problem import DisasterReliefProblem
from core.solution import Solution, Route, ParetoFront
from algorithms.aco import ACO


class SimpleRouteOptimizer:
    """
    Basit rota optimizasyonu (PSO tabanlı)
    """
    
    def __init__(self, problem: DisasterReliefProblem):
        self.problem = problem
    
    def optimize_route(self, depot_id: int, areas: List[int]) -> List[Route]:
        """
        Verilen bölgeler için basit greedy rota oluştur
        (GÜVENLİK YAMALI VERSİYON)
        """
        if len(areas) == 0:
            return []
        
        routes = []
        unvisited = set(areas)
        vehicle_id = 0
        
        depot_node = depot_id
        
        # GÜVENLİK: Sonsuz döngü sayacı
        max_attempts = len(areas) * 2
        attempts = 0
        
        while unvisited:
            attempts += 1
            # Eğer makul deneme sayısını geçersek döngüyü kır
            if attempts > max_attempts:
                print(f"AP UYARI: Sonsuz döngü engellendi. Kalan {len(unvisited)} bölge atlandı.")
                break

            route = Route(depot_id=depot_id, vehicle_id=vehicle_id)
            current_node = depot_node
            current_load = 0.0
            current_time = 0.0
            
            # En yakın bölge stratejisi
            while unvisited:
                # En yakın uygun bölgeyi bul
                next_area = self._find_nearest_feasible(
                    current_node, unvisited, current_load, current_time
                )
                
                if next_area is None:
                    break
                
                # Bölgeyi ekle
                area = self.problem.areas[next_area]
                route.add_area(next_area)
                unvisited.remove(next_area)
                
                # Güncelle
                current_load += area.demand
                area_node = self.problem.num_depots + next_area
                travel_time = self.problem.get_travel_time(current_node, area_node)
                arrival_time = current_time + travel_time
                service_start = max(arrival_time, area.soft_lower)
                current_time = service_start + area.service_time
                current_node = area_node
            
            if not route.is_empty():
                # Rota metriklerini hesapla
                route = self._calculate_metrics(route)
                routes.append(route)
                vehicle_id += 1
            else:
                # KRİTİK EKLENTİ:
                # Yeni kamyon açtık ama hiçbir yere gidemedi.
                # Demek ki kalan müşteriler "imkansız".
                # Sonsuz döngüyü kırmak için birini zorla listeden at.
                if unvisited:
                    unvisited.pop()
        
        return routes
    
    def _find_nearest_feasible(self, current_node, unvisited, 
                               current_load, current_time):
        """En yakın uygun bölgeyi bul"""
        best_area = None
        best_distance = float('inf')
        
        for area_id in unvisited:
            area = self.problem.areas[area_id]
            area_node = self.problem.num_depots + area_id
            
            # Kapasite kontrolü
            if current_load + area.demand > self.problem.vehicle_capacity:
                continue
            
            # Zaman kontrolü
            travel_time = self.problem.get_travel_time(current_node, area_node)
            arrival_time = current_time + travel_time
            service_start = max(arrival_time, area.soft_lower)
            
            if service_start > area.hard_upper:
                continue
            
            # Mesafe
            distance = self.problem.get_distance(current_node, area_node)
            if distance < best_distance:
                best_distance = distance
                best_area = area_id
        
        return best_area
    
    def _calculate_metrics(self, route: Route) -> Route:
        """Rota metriklerini hesapla"""
        problem = self.problem
        depot_node = route.depot_id
        
        route.total_demand = 0.0
        route.total_distance = 0.0
        route.total_time = 0.0
        route.time_window_penalty = 0.0
        
        current_node = depot_node
        current_time = 0.0
        
        for area_id in route.sequence:
            area = problem.areas[area_id]
            area_node = problem.num_depots + area_id
            
            route.total_demand += area.demand
            
            travel_time = problem.get_travel_time(current_node, area_node)
            travel_dist = problem.get_distance(current_node, area_node)
            
            route.total_distance += travel_dist
            route.total_time += travel_time
            
            arrival_time = current_time + travel_time
            service_start = max(arrival_time, area.soft_lower)
            
            penalty = area.get_time_window_penalty(service_start, problem.penalty_rate)
            route.time_window_penalty += penalty
            
            current_time = service_start + area.service_time
            current_node = area_node
        
        # Depoya dön
        return_time = problem.get_travel_time(current_node, depot_node)
        return_dist = problem.get_distance(current_node, depot_node)
        route.total_distance += return_dist
        route.total_time += return_time
        
        return route


class AP:
    """
    Alternative hybrid: ACO for depot assignment, simple greedy for routing
    """
    
    def __init__(self,
                 problem: DisasterReliefProblem,
                 num_iterations: int = 50,
                 num_solutions_per_iter: int = 10):
        
        self.problem = problem
        self.num_iterations = num_iterations
        self.num_solutions = num_solutions_per_iter
        
        # Feromon matrisi (bölge -> depo atama için)
        self.pheromone = {}
        self._initialize_pheromone()
        
        # Rota optimizeri
        self.route_optimizer = SimpleRouteOptimizer(problem)
        
        # Pareto front
        self.pareto_front = ParetoFront()
        
    def _initialize_pheromone(self):
        """Her (bölge, depo) çifti için feromon"""
        for area_id in range(self.problem.num_areas):
            for depot_id in range(self.problem.num_depots):
                self.pheromone[(area_id, depot_id)] = 1.0
    
    def solve(self) -> ParetoFront:
        """Ana çözüm algoritması"""
        print("="*60)
        print("AP Algorithm Started (ACO for assignment + Greedy routing)")
        print("="*60)
        
        for iteration in range(self.num_iterations):
            iteration_solutions = []
            
            # Her iterasyonda birden fazla çözüm üret
            for _ in range(self.num_solutions):
                # ACO ile depo ataması
                assignments = self._construct_assignment()
                
                # Greedy ile rotalama
                solution = self._create_solution(assignments)
                
                iteration_solutions.append(solution)
                self.pareto_front.add(solution)
            
            # Feromon güncelle
            self._update_pheromone(iteration_solutions)
            
            if iteration % 10 == 0:
                best_f1 = min(s.f1_penalty_cost for s in iteration_solutions)
                best_f2 = min(s.f2_operational_cost for s in iteration_solutions)
                print(f"Iteration {iteration}/{self.num_iterations}: "
                      f"f1={best_f1:.2f}, f2={best_f2:.2f}, "
                      f"Pareto={self.pareto_front.size()}")
        
        print("="*60)
        print(f"AP Completed. Pareto size: {self.pareto_front.size()}")
        print("="*60)
        
        return self.pareto_front
    
    def _construct_assignment(self) -> np.ndarray:
        """ACO ile depo ataması oluştur"""
        assignments = np.zeros(self.problem.num_areas, dtype=int)
        
        for area_id in range(self.problem.num_areas):
            # Olasılık hesapla
            probs = []
            for depot_id in range(self.problem.num_depots):
                tau = self.pheromone[(area_id, depot_id)]
                probs.append(tau)
            
            # Normalize
            total = sum(probs)
            probs = [p / total for p in probs]
            
            # Seç
            depot = np.random.choice(self.problem.num_depots, p=probs)
            assignments[area_id] = depot
        
        return assignments
    
    def _create_solution(self, assignments: np.ndarray) -> Solution:
        """Atamalardan tam çözüm oluştur"""
        solution = Solution(self.problem)
        solution.set_depot_assignments(assignments)
        
        all_routes = []
        
        for depot_id in solution.opened_depots:
            assigned = np.where(assignments == depot_id)[0].tolist()
            routes = self.route_optimizer.optimize_route(depot_id, assigned)
            all_routes.extend(routes)
        
        solution.set_routes(all_routes)
        solution.evaluate()
        
        return solution
    
    def _update_pheromone(self, solutions: List[Solution]):
        """Feromon güncelleme"""
        # Buharlaşma
        rho = 0.3
        for key in self.pheromone:
            self.pheromone[key] *= (1 - rho)
        
        # En iyi çözümlerden feromon ekle
        for solution in solutions:
            quality = 100.0 / (1.0 + solution.f2_operational_cost)
            
            for area_id, depot_id in enumerate(solution.depot_assignments):
                self.pheromone[(area_id, depot_id)] += quality