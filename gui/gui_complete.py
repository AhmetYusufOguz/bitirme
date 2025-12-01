"""
GUI ve Visualization modülü - Birleştirilmiş
Tkinter tabanlı arayüz ve matplotlib görselleştirme
"""

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np
from threading import Thread

from core.problem import DisasterReliefProblem
from core.solution import ParetoFront
from algorithms.pa_lrp import PALRP
from algorithms.aco import ACOSolver
from algorithms.pso import PSOSolver
from algorithms.ap import AP
from metrics.metrics_all import AlgorithmComparison


# ============================================================================
# VISUALIZATION.PY - Grafik Çizim Fonksiyonları
# ============================================================================

class Visualizer:
    """Görselleştirme fonksiyonları"""
    
    @staticmethod
    def plot_pareto_front(pareto_fronts: dict, title: str = "Pareto Front Comparison"):
        """
        Birden fazla algoritmanın Pareto frontlarını karşılaştır
        
        Args:
            pareto_fronts: {algorithm_name: ParetoFront} dictionary
            title: Grafik başlığı
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        markers = ['o', 's', '^', 'D']
        
        for idx, (algo_name, front) in enumerate(pareto_fronts.items()):
            objectives = front.get_objectives_array()
            
            if len(objectives) > 0:
                ax.scatter(objectives[:, 0], objectives[:, 1],
                          label=algo_name,
                          color=colors[idx % len(colors)],
                          marker=markers[idx % len(markers)],
                          s=100, alpha=0.7, edgecolors='black', linewidth=0.5)
        
        ax.set_xlabel('f1: Time Window Penalty Cost', fontsize=12)
        ax.set_ylabel('f2: Operational Cost', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        
        return fig
    
    @staticmethod
    def plot_routes_on_map(problem, solution, title: str = "Optimized Routes"):
        """
        Rotaları harita üzerinde göster
        
        Args:
            problem: DisasterReliefProblem
            solution: Solution
            title: Grafik başlığı
        """
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Depoları çiz
        depot_locs = np.array([depot.location for depot in problem.depots])
        ax.scatter(depot_locs[:, 0], depot_locs[:, 1],
                  c='red', marker='s', s=200, 
                  edgecolors='black', linewidth=2,
                  label='Depot', zorder=4)
        
        # Depo etiketleri
        for i, depot in enumerate(problem.depots):
            ax.text(depot.location[0], depot.location[1] + 2,
                   f'D{i+1}', fontsize=10, ha='center',
                   fontweight='bold', color='darkred')
        
        # Bölgeleri çiz
        area_locs = np.array([area.location for area in problem.areas])
        ax.scatter(area_locs[:, 0], area_locs[:, 1],
                  c='blue', s=100,
                  edgecolors='white', linewidth=1,
                  label='Affected Area', zorder=3)
        
        # Bölge etiketleri
        for i, area in enumerate(problem.areas):
            ax.text(area.location[0], area.location[1] + 1.5,
                   str(i+1), fontsize=8, ha='center', color='darkblue')
        
        # Rotaları çiz
        route_colors = plt.cm.rainbow(np.linspace(0, 1, len(solution.routes)))
        
        for route_idx, route in enumerate(solution.routes):
            if route.is_empty():
                continue
            
            depot = problem.depots[route.depot_id]
            
            # Rota çiz
            path_x = [depot.location[0]]
            path_y = [depot.location[1]]
            
            for area_id in route.sequence:
                area = problem.areas[area_id]
                path_x.append(area.location[0])
                path_y.append(area.location[1])
            
            # Depoya dön
            path_x.append(depot.location[0])
            path_y.append(depot.location[1])
            
            ax.plot(path_x, path_y,
                   color=route_colors[route_idx],
                   linewidth=2, alpha=0.6,
                   label=f'Vehicle {route_idx+1} (D{route.depot_id+1})')
        
        ax.set_xlabel('X Coordinate (km)', fontsize=12)
        ax.set_ylabel('Y Coordinate (km)', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='upper right', fontsize=8, ncol=2)
        ax.grid(True, linestyle='--', alpha=0.3)
        plt.tight_layout()
        
        return fig
    
    @staticmethod
    def plot_convergence(convergence_data, title: str = "Convergence History"):
        """
        Yakınsama grafiği
        
        Args:
            convergence_data: Liste of {iteration, best_f1, best_f2, pareto_size}
            title: Grafik başlığı
        """
        fig, axes = plt.subplots(2, 1, figsize=(10, 8))
        
        iterations = [d['iteration'] for d in convergence_data]
        best_f1 = [d['best_f1'] for d in convergence_data]
        best_f2 = [d['best_f2'] for d in convergence_data]
        pareto_sizes = [d['pareto_size'] for d in convergence_data]
        
        # Amaç fonksiyonları
        ax1 = axes[0]
        ax1.plot(iterations, best_f1, 'b-o', label='Best f1 (Penalty)', linewidth=2, markersize=4)
        ax1.plot(iterations, best_f2, 'r-s', label='Best f2 (Cost)', linewidth=2, markersize=4)
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Objective Value')
        ax1.set_title('Best Objective Values per Iteration')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Pareto front boyutu
        ax2 = axes[1]
        ax2.plot(iterations, pareto_sizes, 'g-^', linewidth=2, markersize=4)
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Pareto Front Size')
        ax2.set_title('Pareto Front Growth')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig


# ============================================================================
# MAIN_WINDOW.PY - Ana Arayüz
# ============================================================================

class DisasterReliefGUI:
    """Ana GUI sınıfı"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Disaster Relief Distribution Optimizer")
        self.root.geometry("1400x900")
        
        # Problem ve sonuçlar
        self.problem = None
        self.results = {}
        self.selected_algorithms = []
        
        self._create_ui()
        
    def _create_ui(self):
        """UI bileşenlerini oluştur"""
        # Ana container
        main_container = ttk.Frame(self.root, padding="10")
        main_container.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Sol panel: Parametreler
        left_panel = ttk.LabelFrame(main_container, text="Problem Parameters", padding="10")
        left_panel.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5)
        
        # Sağ panel: Sonuçlar
        right_panel = ttk.LabelFrame(main_container, text="Results", padding="10")
        right_panel.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5)
        
        self._create_parameter_panel(left_panel)
        self._create_results_panel(right_panel)
        
        # Grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_container.columnconfigure(1, weight=2)
        
    def _create_parameter_panel(self, parent):
        """Parametre giriş paneli"""
        row = 0
        
        # Problem boyutu
        ttk.Label(parent, text="Number of Areas:").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.num_areas_var = tk.IntVar(value=50)
        ttk.Entry(parent, textvariable=self.num_areas_var, width=15).grid(row=row, column=1, pady=5)
        row += 1
        
        ttk.Label(parent, text="Number of Depots:").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.num_depots_var = tk.IntVar(value=5)
        ttk.Entry(parent, textvariable=self.num_depots_var, width=15).grid(row=row, column=1, pady=5)
        row += 1
        
        # Araç parametreleri
        ttk.Separator(parent, orient='horizontal').grid(row=row, column=0, columnspan=2, sticky='ew', pady=10)
        row += 1
        
        ttk.Label(parent, text="Vehicle Capacity:").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.vehicle_capacity_var = tk.DoubleVar(value=200)
        ttk.Entry(parent, textvariable=self.vehicle_capacity_var, width=15).grid(row=row, column=1, pady=5)
        row += 1
        
        # Algoritma seçimi
        ttk.Separator(parent, orient='horizontal').grid(row=row, column=0, columnspan=2, sticky='ew', pady=10)
        row += 1
        
        ttk.Label(parent, text="Select Algorithms:", font=('Arial', 10, 'bold')).grid(row=row, column=0, columnspan=2, sticky=tk.W, pady=5)
        row += 1
        
        self.algo_vars = {}
        algorithms = ['PA-LRP', 'PSO', 'ACO', 'AP']
        for algo in algorithms:
            var = tk.BooleanVar(value=(algo == 'PA-LRP'))
            ttk.Checkbutton(parent, text=algo, variable=var).grid(row=row, column=0, columnspan=2, sticky=tk.W)
            self.algo_vars[algo] = var
            row += 1
        
        # Butonlar
        ttk.Separator(parent, orient='horizontal').grid(row=row, column=0, columnspan=2, sticky='ew', pady=10)
        row += 1
        
        ttk.Button(parent, text="Generate Problem", command=self._generate_problem).grid(row=row, column=0, columnspan=2, pady=5)
        row += 1
        
        ttk.Button(parent, text="Run Optimization", command=self._run_optimization).grid(row=row, column=0, columnspan=2, pady=5)
        row += 1
        
        ttk.Button(parent, text="Show Results", command=self._show_results).grid(row=row, column=0, columnspan=2, pady=5)
        row += 1
        
        # Progress
        ttk.Separator(parent, orient='horizontal').grid(row=row, column=0, columnspan=2, sticky='ew', pady=10)
        row += 1
        
        self.progress_var = tk.StringVar(value="Ready")
        ttk.Label(parent, textvariable=self.progress_var, foreground='blue').grid(row=row, column=0, columnspan=2)
        
    def _create_results_panel(self, parent):
        """Sonuç paneli"""
        # Log alanı
        self.log_text = scrolledtext.ScrolledText(parent, width=80, height=40, wrap=tk.WORD)
        self.log_text.pack(fill=tk.BOTH, expand=True)
        
    def _log(self, message):
        """Log mesajı ekle"""
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.root.update()
        
    def _generate_problem(self):
        """Problem örneği oluştur"""
        try:
            num_areas = self.num_areas_var.get()
            num_depots = self.num_depots_var.get()
            vehicle_capacity = self.vehicle_capacity_var.get()
            
            self._log(f"\nGenerating problem: {num_areas} areas, {num_depots} depots...")
            
            self.problem = DisasterReliefProblem.generate_random_instance(
                num_areas=num_areas,
                num_depots=num_depots,
                seed=42
            )
            self.problem.vehicle_capacity = vehicle_capacity
            
            self._log("Problem generated successfully!")
            self._log(f"  - Map size: 100x100 km")
            self._log(f"  - Vehicle capacity: {vehicle_capacity}")
            self._log(f"  - Penalty rate: {self.problem.penalty_rate}")
            
            self.progress_var.set("Problem Ready")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate problem:\n{str(e)}")
    
    def _run_optimization(self):
        """Optimizasyonu çalıştır"""
        if self.problem is None:
            messagebox.showwarning("Warning", "Please generate a problem first!")
            return
        
        # Seçili algoritmaları al
        self.selected_algorithms = [algo for algo, var in self.algo_vars.items() if var.get()]
        
        if not self.selected_algorithms:
            messagebox.showwarning("Warning", "Please select at least one algorithm!")
            return
        
        # Thread'de çalıştır
        thread = Thread(target=self._run_algorithms_thread)
        thread.daemon = True
        thread.start()
    
    def _run_algorithms_thread(self):
        """Algoritmaları ayrı thread'de çalıştır"""
        self.results = {}
        self.progress_var.set("Running...")
        
        for algo_name in self.selected_algorithms:
            self._log(f"\n{'='*60}")
            self._log(f"Running {algo_name}...")
            self._log(f"{'='*60}")
            
            try:
                if algo_name == 'PA-LRP':
                    solver = PALRP(self.problem, num_particles=20, num_pso_iterations=30)
                    result = solver.solve()
                    
                elif algo_name == 'ACO':
                    solver = ACOSolver(self.problem, num_ants=20, num_iterations=30)
                    # ACO için basit wrapper gerekli
                    result = self._run_aco_wrapper(solver)
                    
                elif algo_name == 'PSO':
                    solver = PSOSolver(self.problem, ACOSolver, num_particles=20, num_iterations=30)
                    result = solver.solve()
                    
                elif algo_name == 'AP':
                    solver = AP(self.problem, num_iterations=30)
                    result = solver.solve()
                
                self.results[algo_name] = result
                self._log(f"{algo_name} completed! Pareto size: {result.size()}")
                
            except Exception as e:
                self._log(f"ERROR in {algo_name}: {str(e)}")
        
        self.progress_var.set("Optimization Complete!")
        self._log("\n" + "="*60)
        self._log("All algorithms completed!")
        self._log("="*60)
    
    def _run_aco_wrapper(self, solver):
        """ACO için wrapper (çok sayıda rastgele atama dene)"""
        front = ParetoFront()
        for _ in range(10):
            assignments = np.random.randint(0, self.problem.num_depots, self.problem.num_areas)
            solution = solver.solve(assignments)
            front.add(solution)
        return front
    
    def _show_results(self):
        """Sonuçları göster"""
        if not self.results:
            messagebox.showwarning("Warning", "No results to show! Run optimization first.")
            return
        
        # Pareto front karşılaştırması
        Visualizer.plot_pareto_front(self.results, "Pareto Front Comparison")
        plt.show()
        
        # Metrik karşılaştırması
        comparison = AlgorithmComparison()
        for algo_name, front in self.results.items():
            comparison.add_result(algo_name, front)
        
        self._log("\n" + "="*60)
        self._log("PERFORMANCE METRICS")
        self._log("="*60)
        
        metrics_dict = comparison.compare_all()
        
        # Başlık
        self._log(f"{'Algorithm':<15} {'IGD':<12} {'HV':<12} {'QM':<8} {'SM':<10} {'Size':<6}")
        self._log("-"*60)
        
        for algo_name, metrics in metrics_dict.items():
            self._log(f"{algo_name:<15} "
                     f"{metrics['IGD']:<12.4f} "
                     f"{metrics['HV']:<12.2f} "
                     f"{metrics['QM']:<8.4f} "
                     f"{metrics['SM']:<10.4f} "
                     f"{metrics['Size']:<6}")
        
        self._log("="*60)
        
        # En iyi çözümün haritasını göster (PA-LRP'den)
        if 'PA-LRP' in self.results:
            best_solution = self.results['PA-LRP'].get_solutions()[0]
            Visualizer.plot_routes_on_map(self.problem, best_solution, "PA-LRP Best Solution")
            plt.show()
    
    def run(self):
        """GUI'yi başlat"""
        self.root.mainloop()


# ============================================================================
# Ana Çalıştırma
# ============================================================================

if __name__ == "__main__":
    app = DisasterReliefGUI()
    app.run()