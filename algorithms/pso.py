"""
Particle Swarm Optimization (PSO) Algorithm
Depo atama optimizasyonu için kullanılır
Makaledeki Section 3.4'ü uygular
"""

import numpy as np
from typing import List, Tuple
from core.problem import DisasterReliefProblem
from core.solution import Solution, ParetoFront


class Particle:
    """
    Bir parçacık (depo atama planı)
    """
    
    def __init__(self, num_areas: int, num_depots: int):
        self.num_areas = num_areas
        self.num_depots = num_depots
        
        # Pozisyon: her bölge için depo ataması (sürekli değerler)
        self.position = np.random.rand(num_areas) * num_depots
        
        # Hız
        self.velocity = np.random.randn(num_areas) * 0.1
        
        # En iyi kişisel pozisyon
        self.personal_best_position = self.position.copy()
        self.personal_best_fitness = (float('inf'), float('inf'))
        
        # Mevcut fitness
        self.current_fitness = (float('inf'), float('inf'))
    
    def get_depot_assignments(self) -> np.ndarray:
        """
        Sürekli pozisyonu discrete depo atamalarına dönüştür
        Makaledeki Figures 3-4 örneği
        """
        # Her değeri [0, num_depots) aralığına sınırla
        bounded = np.clip(self.position, 0, self.num_depots - 0.001)
        
        # Tam sayıya yuvarla
        assignments = np.floor(bounded).astype(int)
        
        return assignments
    
    def update_velocity(self, 
                       global_best_position: np.ndarray,
                       w: float = 1.0,
                       c1: float = 2.0,
                       c2: float = 2.0):
        """
        Hızı güncelle (Equation 21)
        
        Args:
            global_best_position: En iyi global pozisyon
            w: Atalet katsayısı (weight)
            c1: Bilişsel katsayı (kişisel en iyi)
            c2: Sosyal katsayı (global en iyi)
        """
        r1 = np.random.rand(self.num_areas)
        r2 = np.random.rand(self.num_areas)
        
        # Equation 21
        cognitive = c1 * r1 * (self.personal_best_position - self.position)
        social = c2 * r2 * (global_best_position - self.position)
        
        self.velocity = w * self.velocity + cognitive + social
        
        # Hız sınırlaması
        max_velocity = 2.0
        self.velocity = np.clip(self.velocity, -max_velocity, max_velocity)
    
    def update_position(self):
        """
        Pozisyonu güncelle (Equation 22)
        """
        self.position = self.position + self.velocity
        
        # Pozisyonu [0, num_depots] aralığında tut
        self.position = np.clip(self.position, 0, self.num_depots)


class PSO:
    """
    Particle Swarm Optimization
    Depo atama için kullanılır
    """
    
    def __init__(self,
                 problem: DisasterReliefProblem,
                 num_particles: int = 30,
                 num_iterations: int = 50,
                 w: float = 1.0,      # Atalet
                 c1: float = 2.0,     # Bilişsel
                 c2: float = 2.0):    # Sosyal
        
        self.problem = problem
        self.num_particles = num_particles
        self.num_iterations = num_iterations
        self.w = w
        self.c1 = c1
        self.c2 = c2
        
        # Parçacıklar
        self.particles: List[Particle] = []
        
        # Global en iyi (Pareto frontu)
        self.pareto_front = ParetoFront()
        
        # En iyi global pozisyon (crowding distance'a göre)
        self.global_best_position = None
        
    def initialize_particles(self):
        """Parçacıkları başlat"""
        self.particles = []
        for _ in range(self.num_particles):
            particle = Particle(self.problem.num_areas, self.problem.num_depots)
            self.particles.append(particle)
    
    def optimize(self, route_optimizer) -> ParetoFront:
        """
        PSO optimizasyonu
        
        Args:
            route_optimizer: Rota optimizasyonu için kullanılacak nesne
                           (ACO veya başka bir yöntem)
        
        Returns:
            Pareto optimal çözümler
        """
        self.initialize_particles()
        
        # İlk global en iyi pozisyonu rastgele seç
        self.global_best_position = self.particles[0].position.copy()
        
        for iteration in range(self.num_iterations):
            # Her parçacık için
            for particle in self.particles:
                # Depo atamalarını al
                assignments = particle.get_depot_assignments()
                
                # Bu atamalar için rotaları optimize et
                solution = route_optimizer.solve(assignments)
                
                # Fitness değerlerini al
                f1, f2 = solution.get_objectives()
                particle.current_fitness = (f1, f2)
                
                # Kişisel en iyiyi güncelle
                if self._dominates(particle.current_fitness, particle.personal_best_fitness):
                    particle.personal_best_position = particle.position.copy()
                    particle.personal_best_fitness = particle.current_fitness
                
                # Pareto frontuna ekle
                self.pareto_front.add(solution)
            
            # Global en iyiyi güncelle (Pareto frontundan seç)
            self._update_global_best()
            
            # Tüm parçacıkların hız ve pozisyonlarını güncelle
            for particle in self.particles:
                particle.update_velocity(self.global_best_position, 
                                        self.w, self.c1, self.c2)
                particle.update_position()
            
            if iteration % 10 == 0:
                print(f"PSO Iteration {iteration}/{self.num_iterations}, "
                      f"Pareto size: {self.pareto_front.size()}")
        
        return self.pareto_front
    
    def _dominates(self, fitness1: Tuple[float, float], 
                   fitness2: Tuple[float, float]) -> bool:
        """Bir fitness değeri diğerini domine ediyor mu?"""
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
        # Çözümden depo atamalarını al
        if best_solution.depot_assignments is not None:
            self.global_best_position = best_solution.depot_assignments.astype(float)
        else:
            # Fallback: random position
            self.global_best_position = np.random.rand(self.problem.num_areas) * self.problem.num_depots
    
    def _calculate_crowding_distances(self, solutions: List[Solution]) -> np.ndarray:
        """
        Crowding distance hesapla
        Çeşitlilik için kullanılır
        """
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


class PSOSolver:
    """
    Tam PSO çözücü (rota optimizasyonu ile birlikte)
    """
    
    def __init__(self, 
                 problem: DisasterReliefProblem,
                 route_optimizer_class,
                 **pso_params):
        self.problem = problem
        self.pso = PSO(problem, **pso_params)
        self.route_optimizer = route_optimizer_class(problem)
    
    def solve(self) -> ParetoFront:
        """
        Tam problemi çöz
        """
        return self.pso.optimize(self.route_optimizer)