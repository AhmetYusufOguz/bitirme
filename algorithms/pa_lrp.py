"""
Particle-Assisted ACO (PA-LRP) Algorithm
Makaledeki ana hibrit algoritma (Fig. 1 ve Section 3)
PSO depo ataması yapar, ACO rotaları optimize eder
"""

import numpy as np
from typing import List
from core.problem import DisasterReliefProblem
from core.solution import Solution, ParetoFront
from algorithms.pso import PSO, Particle
from algorithms.aco import ACO


class PALRP:
    """
    Particle-Assisted Location-Routing Problem Solver
    
    Makaledeki hibrit yaklaşım:
    1. PSO parçacıkları depo atamalarını optimize eder
    2. Her atama için ACO rotaları optimize eder
    3. İki amaç fonksiyonu için Pareto frontu oluşturur
    """
    
    def __init__(self,
                 problem: DisasterReliefProblem,
                 # PSO parametreleri
                 num_particles: int = 30,
                 num_pso_iterations: int = 50,
                 pso_w: float = 1.0,
                 pso_c1: float = 2.0,
                 pso_c2: float = 2.0,
                 # ACO parametreleri
                 num_ants: int = 30,
                 num_aco_iterations: int = 20,
                 aco_alpha: float = 1.0,
                 aco_beta: float = 0.0,
                 aco_rho: float = 0.3,
                 aco_q: float = 100.0):
        
        self.problem = problem
        
        # PSO için parçacıklar oluştur
        self.pso = PSO(
            problem=problem,
            num_particles=num_particles,
            num_iterations=num_pso_iterations,
            w=pso_w,
            c1=pso_c1,
            c2=pso_c2
        )
        
        # ACO parametrelerini sakla
        self.aco_params = {
            'num_ants': num_ants,
            'num_iterations': num_aco_iterations,
            'alpha': aco_alpha,
            'beta': aco_beta,
            'rho': aco_rho,
            'q': aco_q
        }
        
        # Pareto optimal çözümler
        self.pareto_front = ParetoFront()
        
        # İstatistikler
        self.convergence_history = []
        
    def solve(self) -> ParetoFront:
        """
        Ana çözüm algoritması
        Makaledeki Fig. 1'i takip eder
        
        Returns:
            Pareto optimal çözümler kümesi
        """
        print("="*60)
        print("PA-LRP Algorithm Started")
        print(f"Problem: {self.problem.num_areas} areas, "
              f"{self.problem.num_depots} depots")
        print("="*60)
        
        # PSO parçacıklarını başlat
        self.pso.initialize_particles()
        
        # İlk global en iyi pozisyonu ayarla
        self.pso.global_best_position = self.pso.particles[0].position.copy()
        
        # Ana optimizasyon döngüsü
        for iteration in range(self.pso.num_iterations):
            print(f"\n--- PSO Iteration {iteration + 1}/{self.pso.num_iterations} ---")
            
            iteration_best_f1 = float('inf')
            iteration_best_f2 = float('inf')
            
            # Her parçacık için
            for particle_idx, particle in enumerate(self.pso.particles):
                # 1. Depo atamalarını al (PSO pozisyonundan)
                depot_assignments = particle.get_depot_assignments()
                
                # 2. Bu atamalar için ACO ile rotaları optimize et
                solution = self._optimize_routes_with_aco(depot_assignments)
                
                # 3. Fitness değerlerini al
                f1, f2 = solution.get_objectives()
                particle.current_fitness = (f1, f2)
                
                # İterasyondaki en iyi değerleri takip et
                if f1 < iteration_best_f1:
                    iteration_best_f1 = f1
                if f2 < iteration_best_f2:
                    iteration_best_f2 = f2
                
                # 4. Kişisel en iyiyi güncelle
                if self._dominates(particle.current_fitness, 
                                  particle.personal_best_fitness):
                    particle.personal_best_position = particle.position.copy()
                    particle.personal_best_fitness = particle.current_fitness
                
                # 5. Pareto frontuna ekle
                self.pareto_front.add(solution)
            
            # 6. Global en iyiyi güncelle
            self._update_global_best()
            
            # 7. Tüm parçacıkların hız ve pozisyonlarını güncelle
            for particle in self.pso.particles:
                particle.update_velocity(
                    self.pso.global_best_position,
                    self.pso.w, self.pso.c1, self.pso.c2
                )
                particle.update_position()
            
            # İstatistikleri kaydet
            self.convergence_history.append({
                'iteration': iteration + 1,
                'pareto_size': self.pareto_front.size(),
                'best_f1': iteration_best_f1,
                'best_f2': iteration_best_f2
            })
            
            # İlerleme raporu
            print(f"  Best f1 (penalty): {iteration_best_f1:.2f}")
            print(f"  Best f2 (cost): {iteration_best_f2:.2f}")
            print(f"  Pareto front size: {self.pareto_front.size()}")
        
        print("\n" + "="*60)
        print("PA-LRP Algorithm Completed")
        print(f"Final Pareto Front Size: {self.pareto_front.size()}")
        print("="*60)
        
        return self.pareto_front
    
    def _optimize_routes_with_aco(self, 
                                   depot_assignments: np.ndarray) -> Solution:
        """
        Verilen depo atamalarına göre ACO ile rotaları optimize et
        Makaledeki Table 1 algoritması
        """
        # Her çalışma için yeni ACO örneği
        aco = ACO(self.problem, **self.aco_params)
        
        # Çözüm oluştur
        solution = Solution(self.problem)
        solution.set_depot_assignments(depot_assignments)
        
        all_routes = []
        
        # Her açık depo için rotaları optimize et
        for depot_id in solution.opened_depots:
            # Bu depoya atanan bölgeleri bul
            assigned_areas = np.where(depot_assignments == depot_id)[0].tolist()
            
            if len(assigned_areas) == 0:
                continue
            
            # ACO ile bu depo için rotaları bul
            routes = aco.optimize_routes_for_depot(depot_id, assigned_areas)
            
            # Rota metriklerini hesapla
            for route in routes:
                route = solution.calculate_route_metrics(route)
            
            all_routes.extend(routes)
        
        # Rotaları çözüme ekle ve değerlendir
        solution.set_routes(all_routes)
        solution.evaluate()
        
        return solution
    
    def _dominates(self, fitness1, fitness2) -> bool:
        """Pareto dominans kontrolü"""
        better_in_one = (fitness1[0] < fitness2[0] or fitness1[1] < fitness2[1])
        not_worse = (fitness1[0] <= fitness2[0] and fitness1[1] <= fitness2[1])
        return not_worse and better_in_one
    
    def _update_global_best(self):
        """
        Global en iyi pozisyonu güncelle
        Pareto frontundan maksimum crowding distance'a sahip çözümü seç
        """
        if self.pareto_front.size() == 0:
            return
        
        solutions = self.pareto_front.get_solutions()
        
        # Crowding distance hesapla
        crowding_distances = self._calculate_crowding_distances(solutions)
        
        # Maksimum crowding distance'a sahip çözümü seç
        max_idx = np.argmax(crowding_distances)
        best_solution = solutions[max_idx]
        
        # Global best pozisyonu güncelle
        self.pso.global_best_position = best_solution.depot_assignments.astype(float)
    
    def _calculate_crowding_distances(self, solutions: List[Solution]) -> np.ndarray:
        """Crowding distance hesapla (çeşitlilik için)"""
        n = len(solutions)
        if n <= 2:
            return np.full(n, float('inf'))
        
        distances = np.zeros(n)
        
        # Amaç fonksiyonları
        objectives = np.array([[s.f1_penalty_cost, s.f2_operational_cost] 
                              for s in solutions])
        
        # Her amaç için
        for obj_idx in range(2):
            # Bu amaca göre sırala
            sorted_indices = np.argsort(objectives[:, obj_idx])
            
            # Uç noktalara sonsuz mesafe
            distances[sorted_indices[0]] = float('inf')
            distances[sorted_indices[-1]] = float('inf')
            
            # Normalizasyon için aralık
            obj_range = objectives[sorted_indices[-1], obj_idx] - \
                       objectives[sorted_indices[0], obj_idx]
            
            if obj_range == 0:
                continue
            
            # Ara noktalar için crowding distance
            for i in range(1, n - 1):
                idx = sorted_indices[i]
                idx_prev = sorted_indices[i - 1]
                idx_next = sorted_indices[i + 1]
                
                distance = (objectives[idx_next, obj_idx] - 
                          objectives[idx_prev, obj_idx]) / obj_range
                distances[idx] += distance
        
        return distances
    
    def get_convergence_data(self):
        """Yakınsama verilerini döndür (grafik için)"""
        return self.convergence_history
    
    def get_best_solution_by_preference(self, 
                                       weight_f1: float = 0.5,
                                       weight_f2: float = 0.5) -> Solution:
        """
        Kullanıcı tercihlerine göre en iyi çözümü seç
        
        Args:
            weight_f1: Zaman penceresi cezası için ağırlık
            weight_f2: Operasyonel maliyet için ağırlık
            
        Returns:
            Ağırlıklı skora göre en iyi çözüm
        """
        solutions = self.pareto_front.get_solutions()
        if len(solutions) == 0:
            return None
        
        # Normalize et
        f1_values = [s.f1_penalty_cost for s in solutions]
        f2_values = [s.f2_operational_cost for s in solutions]
        
        f1_min, f1_max = min(f1_values), max(f1_values)
        f2_min, f2_max = min(f2_values), max(f2_values)
        
        f1_range = f1_max - f1_min if f1_max > f1_min else 1.0
        f2_range = f2_max - f2_min if f2_max > f2_min else 1.0
        
        best_solution = None
        best_score = float('inf')
        
        for solution in solutions:
            # Normalize edilmiş değerler
            norm_f1 = (solution.f1_penalty_cost - f1_min) / f1_range
            norm_f2 = (solution.f2_operational_cost - f2_min) / f2_range
            
            # Ağırlıklı skor
            score = weight_f1 * norm_f1 + weight_f2 * norm_f2
            
            if score < best_score:
                best_score = score
                best_solution = solution
        
        return best_solution