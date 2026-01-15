"""
GUI ve Visualization modülü - Birleştirilmiş
Tkinter tabanlı arayüz ve matplotlib görselleştirme
Includes Dashboard-style Map and Schedule Views
"""

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import matplotlib.pyplot as plt
import numpy as np
from threading import Thread

# Backend Imports
from core.problem import DisasterReliefProblem
from core.solution import ParetoFront
from algorithms.pa_lrp import PALRP
from algorithms. aco import ACOSolver
from algorithms.pso import PSOSolver
from algorithms.ap import AP
from metrics.metrics_all import AlgorithmComparison

# ============================================================================
# VISUALIZATION. PY - Grafik Çizim Fonksiyonları
# ============================================================================

class Visualizer:
    """Görselleştirme fonksiyonları"""
    
    @staticmethod
    def plot_pareto_front(pareto_fronts:  dict, title: str = "Pareto Front Comparison"):
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        markers = ['o', 's', '^', 'D']
        
        for idx, (algo_name, front) in enumerate(pareto_fronts.items()):
            objectives = front.get_objectives_array()
            if len(objectives) > 0:
                ax.scatter(objectives[: , 0], objectives[:, 1],
                          label=algo_name, color=colors[idx % len(colors)],
                          marker=markers[idx % len(markers)], s=100, alpha=0.7, edgecolors='black')
        
        ax.set_xlabel('f1: Time Window Penalty Cost')
        ax.set_ylabel('f2: Operational Cost')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_dashboard_map(problem, solution, title: str = "Optimized Relief Routes"):
        """
        Creates the 'Professional' Square Map View (No Stretching)
        """
        fig, ax_map = plt.subplots(figsize=(10, 10))
        
        # Data Preparation
        depot_locs = np.array([d.location for d in problem.depots])
        area_locs = np.array([a.location for a in problem.areas])
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

        # 1. Draw Routes
        if solution.routes:
            # Sort for consistent coloring
            sorted_routes = sorted(solution.routes, key=lambda r: r.depot_id)
            for i, route in enumerate(sorted_routes):
                if route.is_empty(): continue
                
                # Reconstruct full path
                path_x = [depot_locs[route.depot_id][0]] + \
                         [area_locs[aid][0] for aid in route. sequence] + \
                         [depot_locs[route.depot_id][0]]
                path_y = [depot_locs[route.depot_id][1]] + \
                         [area_locs[aid][1] for aid in route.sequence] + \
                         [depot_locs[route.depot_id][1]]
                
                ax_map. plot(path_x, path_y, c=colors[i % len(colors)], linewidth=2, alpha=0.8, label=f'Vehicle {i+1}')

        # 2. Draw Areas
        ax_map.scatter(area_locs[:, 0], area_locs[:, 1], c='blue', s=80, zorder=3, edgecolors='white')
        for i, area in enumerate(problem.areas):
             ax_map.text(area.location[0], area.location[1] + 1.5, str(i), fontsize=8, ha='center', color='darkblue')

        # 3. Draw Depots
        opened_ids = set(solution.opened_depots)
        for i, depot in enumerate(problem.depots):
            color = 'red' if i in opened_ids else 'lightgray'
            size = 200 if i in opened_ids else 50
            marker = 's' if i in opened_ids else 'x'
            zorder = 4 if i in opened_ids else 2
            
            ax_map.scatter(depot.location[0], depot.location[1], c=color, marker=marker, s=size, zorder=zorder, edgecolors='black')
            ax_map.text(depot.location[0], depot.location[1]+3, f'D{i}', fontsize=11, ha='center', fontweight='bold', color='darkred')

        # 4. Styling (Square Aspect Ratio to prevent stretching)
        ax_map.set_aspect('equal')
        ax_map.set_title(title, fontsize=14, fontweight='bold')
        ax_map.grid(True, linestyle='--', alpha=0.4)
        ax_map.set_xlim(-5, 105)
        ax_map.set_ylim(-5, 105)
        ax_map.set_xlabel("X Coordinate (km)")
        ax_map.set_ylabel("Y Coordinate (km)")
        
        plt.tight_layout()
        return fig

    @staticmethod
    def generate_schedule_text(problem, solution):
        """Generates the detailed formatted text log for the schedule window"""
        
        # --- YENİ EKLENEN KISIM: ALGORİTMA VE ÇÖZÜM BİLGİSİ ---
        # Eğer çözüm nesnesinde etiket varsa onu al, yoksa varsayılan yaz
        algo_name = getattr(solution, 'algo_name', 'Unknown')
        pareto_count = getattr(solution, 'pareto_size_found', '?')

        log_text = "OPTIMIZED DELIVERY SCHEDULE\n"
        log_text += f"ALGORITHM: {algo_name}\n"
        log_text += f"SOLUTIONS FOUND: {pareto_count}\n"
        log_text += "="*60 + "\n"
        # -----------------------------------------------------
        
        if not solution.routes:
            return log_text + "\nNo feasible solution found."

        # Yazım hatası düzeltildi: 'r. depot_id' -> 'r.depot_id'
        sorted_routes = sorted(solution.routes, key=lambda r: r.depot_id)
        
        for i, route in enumerate(sorted_routes):
            if route.is_empty(): continue
            
            # Route Header
            log_text += f"\nVEHICLE {i+1} [Depot {route.depot_id}]\n"
            log_text += f"  Stats: {route.total_distance:.1f}km | Load: {int(route.total_demand)}/{int(problem.vehicle_capacity)}\n"
            log_text += f"  {'Stop': <6} | {'Arr':<8} | {'Window':<15} | {'Status'}\n"
            log_text += "  " + "-"*50 + "\n"
            
            # Re-simulate time to get precise arrival details
            # Senin kodundaki özel başlangıç saati
            current_time = 20 
            curr_loc = problem.depots[route.depot_id].location
            
            for area_id in route.sequence:
                area = problem.areas[area_id]
                target_loc = area.location
                
                # Calc distance & time
                dist = np.sqrt(np.sum((np.array(target_loc) - np.array(curr_loc))**2))
                # Senin kodundaki özel hız mantığı (1.5 min per km)
                travel_time = int(dist * 1.5) 
                arrival = current_time + travel_time
                
                # Check Windows
                e_win = int(area.soft_lower)
                l_win = int(area.soft_upper)
                
                status = ""
                if arrival < e_win:
                    wait = e_win - arrival
                    status = f"WAIT ({wait}m)"
                    start_service = e_win
                elif arrival <= l_win:
                    status = "ON TIME"
                    start_service = arrival
                else:
                    late = arrival - l_win
                    status = f"LATE (+{late}m)"
                    start_service = arrival
                
                # Format string
                arr_s = f"{arrival//60:02d}:{arrival%60:02d}"
                win_s = f"[{e_win//60:02d}-{l_win//60:02d}]"
                
                log_text += f"  Area {area_id: <2} | {arr_s: <8} | {win_s: <15} | {status}\n"
                
                # Yazım hatası düzeltildi: 'area. service_time' -> 'area.service_time'
                current_time = start_service + int(area.service_time)
                curr_loc = target_loc
                
        log_text += "\n" + "="*60 + "\n"
        log_text += f"Active Vehicles:   {len(solution.routes)}\n"
        log_text += f"Total Distance:   {solution.transport_cost / problem.transport_cost_rate:.1f} km\n"
        log_text += f"Operational Cost: {solution.f2_operational_cost:.2f}"
        
        return log_text
# ============================================================================
# MAIN_WINDOW.PY - Ana Arayüz
# ============================================================================
class DisasterReliefGUI: 
    """Ana GUI sınıfı"""
    
    def __init__(self):
        self.root = tk. Tk()
        self.root.title("Disaster Relief Distribution Optimizer")
        self.root.geometry("1400x900")
        
        self.problem = None
        self. results = {}
        self.selected_algorithms = []
        
        self._create_ui()
        
    def _create_ui(self):
        main_container = ttk.Frame(self.root, padding="10")
        main_container.grid(row=0, column=0, sticky=(tk.W, tk. E, tk.N, tk. S))
        
        # Left Panel (Parameters)
        left_panel = ttk.LabelFrame(main_container, text="Control Panel", padding="10")
        left_panel.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5)
        
        # Right Panel (Log)
        right_panel = ttk.LabelFrame(main_container, text="System Log", padding="10")
        right_panel.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5)
        
        self._create_parameter_panel(left_panel)
        self._create_results_panel(right_panel)
        
        self.root. columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_container.columnconfigure(1, weight=2)
        
    def _create_parameter_panel(self, parent):
        row = 0
        
        # --- Problem Generation ---
        ttk. Label(parent, text="PROBLEM SETTINGS", font=('Arial', 9, 'bold')).grid(row=row, column=0, columnspan=2, pady=5)
        row += 1
        
        # Areas Input
        ttk.Label(parent, text="Areas:").grid(row=row, column=0, sticky=tk.W)
        self.num_areas_var = tk.IntVar(value=50)
        ttk.Entry(parent, textvariable=self.num_areas_var, width=10).grid(row=row, column=1)
        row += 1
        
        # Depots Input
        ttk.Label(parent, text="Depots:").grid(row=row, column=0, sticky=tk.W)
        self.num_depots_var = tk.IntVar(value=5)
        ttk.Entry(parent, textvariable=self.num_depots_var, width=10).grid(row=row, column=1)
        row += 1

        # Vehicle Capacity Input
        ttk.Label(parent, text="Veh. Cap:").grid(row=row, column=0, sticky=tk. W)
        self.vehicle_capacity_var = tk. DoubleVar(value=200)
        ttk.Entry(parent, textvariable=self. vehicle_capacity_var, width=10).grid(row=row, column=1)
        row += 1
        
        # Generate Problem Button
        ttk.Button(parent, text="1. Generate Problem", command=self._generate_problem).grid(row=row, column=0, columnspan=2, pady=5, sticky="ew")
        row += 1
        
        # *** YENİ:  Load Fixed Test Button ***
        ttk.Button(parent, text="Load Fixed Test", 
                  command=self._load_fixed_test, 
                  style="Accent.TButton").grid(row=row, column=0, columnspan=2, pady=5, sticky="ew")
        row += 1
        
        ttk. Separator(parent, orient='horizontal').grid(row=row, column=0, columnspan=2, sticky='ew', pady=10)
        row += 1

        # --- Algorithms ---
        ttk.Label(parent, text="ALGORITHMS", font=('Arial', 9, 'bold')).grid(row=row, column=0, columnspan=2, pady=5)
        row += 1
        
        self.algo_vars = {}
        algorithms = ['PA-LRP', 'PSO', 'ACO', 'AP']
        for algo in algorithms: 
            var = tk.BooleanVar(value=(algo == 'PA-LRP'))
            ttk.Checkbutton(parent, text=algo, variable=var).grid(row=row, column=0, columnspan=2, sticky=tk.W)
            self.algo_vars[algo] = var
            row += 1
            
        ttk.Button(parent, text="2. Run Optimization", command=self._run_optimization).grid(row=row, column=0, columnspan=2, pady=10, sticky="ew")
        row += 1
        
        ttk.Separator(parent, orient='horizontal').grid(row=row, column=0, columnspan=2, sticky='ew', pady=10)
        row += 1
        
        # --- Visualization Buttons ---
        ttk.Label(parent, text="VISUALIZATION", font=('Arial', 9, 'bold')).grid(row=row, column=0, columnspan=2, pady=5)
        row += 1
        
        ttk.Button(parent, text="3. Compare Pareto Fronts", command=self._show_results).grid(row=row, column=0, columnspan=2, pady=2, sticky="ew")
        row += 1
        
        ttk.Button(parent, text="4. Show Route Map", command=self._show_route_map_window).grid(row=row, column=0, columnspan=2, pady=2, sticky="ew")
        row += 1
        
        ttk.Button(parent, text="5. Show Schedule Log", command=self._show_schedule_window).grid(row=row, column=0, columnspan=2, pady=2, sticky="ew")
        row += 1
        
        # Status Label
        self.progress_var = tk.StringVar(value="Ready")
        ttk.Label(parent, textvariable=self.progress_var, foreground='blue', font=('Arial', 8)).grid(row=row, column=0, columnspan=2, pady=10)
        
    def _create_results_panel(self, parent):
        self.log_text = scrolledtext.ScrolledText(parent, width=60, height=40, wrap=tk.WORD)
        self.log_text. pack(fill=tk.BOTH, expand=True)
        
    def _log(self, message):
        self.log_text.insert(tk. END, message + "\n")
        self.log_text.see(tk.END)
        self.root.update()
        
    def _generate_problem(self):
        try:
            area_count = self.num_areas_var.get()
            depot_count = self.num_depots_var.get()
            veh_cap = self.vehicle_capacity_var.get()

            self._log(f"\nGenerating problem:  {area_count} Areas, {depot_count} Depots, Cap {veh_cap}...")
            
            self.problem = DisasterReliefProblem. generate_random_instance(
                num_areas=area_count,
                num_depots=depot_count,
                seed=42
            )
            
            self. problem.vehicle_capacity = veh_cap
            
            # Keep the safety buffer for Depots ONLY
            for d in self. problem.depots: 
                d.capacity = 300 
            
            self._log("Problem Generated Successfully!")
            self. progress_var.set("Problem Ready")
        except Exception as e:
            messagebox.showerror("Error", str(e))
    
    # *** YENİ FONKSİYON:  Sabit Test Yükle ***
    def _load_fixed_test(self):
        """Sabit test senaryosu:  20 area, 3 depot, 75 kapasite"""
        try:
            self._log("\n" + "="*60)
            self._log("Loading FIXED TEST Scenario...")
            self._log("Parameters:  20 Areas, 3 Depots, Vehicle Cap:  75")
            self._log("="*60)
            
            # Sabit parametreler
            fixed_areas = 20
            fixed_depots = 3
            fixed_capacity = 75
            fixed_seed = 42
            
            # GUI değerlerini güncelle
            self.num_areas_var.set(fixed_areas)
            self.num_depots_var.set(fixed_depots)
            self.vehicle_capacity_var.set(fixed_capacity)
            
            # Problem oluştur (sabit seed ile)
            self.problem = DisasterReliefProblem. generate_random_instance(
                num_areas=fixed_areas,
                num_depots=fixed_depots,
                seed=fixed_seed
            )
            
            self. problem.vehicle_capacity = fixed_capacity
            
            # Depot kapasitelerini ayarla
            for d in self.problem.depots: 
                d.capacity = 300
            
            self._log("\n✅ Fixed Test Loaded Successfully!")
            self._log(f"   - Areas: {fixed_areas}")
            self._log(f"   - Depots: {fixed_depots}")
            self._log(f"   - Vehicle Capacity: {fixed_capacity}")
            self._log(f"   - Seed: {fixed_seed}")
            self._log("\nYou can now run optimization and compare results!")
            
            self.progress_var.set("Fixed Test Ready")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load fixed test: {str(e)}")
            self._log(f"❌ Error: {str(e)}")
    
    def _run_optimization(self):
        if self.problem is None:
            messagebox.showwarning("Warning", "Generate problem first!")
            return
        
        self. selected_algorithms = [algo for algo, var in self.algo_vars. items() if var.get()]
        if not self.selected_algorithms:
            messagebox.showwarning("Warning", "Select an algorithm!")
            return
        
        Thread(target=self._run_algorithms_thread, daemon=True).start()
    
    def _run_algorithms_thread(self):
        """Akıllı Kıyaslama: Maliyet yakınsa Çözüm Sayısına bakar"""
        self.results = {}
        self.progress_var.set("Running...")

        comparison = AlgorithmComparison()
        
        # Takip değişkenleri
        global_best_solution = None
        global_best_cost = float('inf')
        global_best_pareto_size = 0
        global_best_algo_name = ""

        for algo_name in self.selected_algorithms:
            self._log(f"\n{'='*60}")
            self._log(f"Running {algo_name}...")
            self._log(f"{'='*60}")
            
            try:
                # --- ALGORİTMA ÇALIŞTIRMA KISMI (AYNI) ---
                if algo_name == 'PA-LRP':
                    solver = PALRP(self.problem, num_particles=20, num_pso_iterations=30)
                    result = solver.solve()
                elif algo_name == 'ACO':
                    solver = ACOSolver(self.problem, num_ants=20, num_iterations=30)
                    result = self._run_aco_wrapper(solver)
                elif algo_name == 'PSO':
                    solver = PSOSolver(self.problem, ACOSolver, num_particles=20, num_iterations=30)
                    result = solver.solve()
                elif algo_name == 'AP':
                    solver = AP(self.problem, num_iterations=30)
                    result = solver.solve()
                
                self.results[algo_name] = result
                comparison.add_result(algo_name, result)
                
                if result.size() > 0:
                    solutions = result.get_solutions()
                    # Bu algoritmanın en iyi maliyetli çözümünü bul
                    best_sol_of_algo = min(solutions, key=lambda s: s.f2_operational_cost)
                    
                    # --- ÇÖZÜME ETİKET YAPIŞTIR (Logda göstermek için) ---
                    best_sol_of_algo.algo_name = algo_name
                    best_sol_of_algo.pareto_size_found = result.size()
                    
                    # İstatistikler
                    active_vehicles = len(best_sol_of_algo.routes)
                    cost_rate = getattr(self.problem, 'transport_cost_rate', 0.01) 
                    total_dist = best_sol_of_algo.transport_cost / cost_rate
                    op_cost = best_sol_of_algo.f2_operational_cost
                    pareto_size = result.size()
                    
                    self._log("-" * 40)
                    self._log(f"  [BEST RESULT FOR {algo_name}]")
                    self._log(f"  > Solutions Found : {pareto_size}")
                    self._log(f"  > Active Vehicles : {active_vehicles}")
                    self._log(f"  > Total Distance  : {total_dist:.1f} km")
                    self._log(f"  > Operational Cost: {op_cost:.2f}")
                    self._log("-" * 40)

                    # --- AKILLI KIYASLAMA MANTIĞI ---
                    is_new_winner = False
                    
                    # Durum 1: Henüz hiç lider yoksa
                    if global_best_solution is None:
                        is_new_winner = True
                    
                    # Durum 2: Yeni çözüm BELİRGİN şekilde daha ucuzsa (Fark > 0.5)
                    elif (global_best_cost - op_cost) > 0.5:
                        is_new_winner = True
                        print(f"   -> {algo_name} took lead by COST ({op_cost:.2f} < {global_best_cost:.2f})")
                        
                    # Durum 3: Maliyetler ÇOK YAKINSA (Fark < 0.5), PARETO SIZE'a bak
                    elif abs(global_best_cost - op_cost) <= 0.5:
                        if pareto_size > global_best_pareto_size:
                            is_new_winner = True
                            print(f"   -> {algo_name} took lead by DIVERSITY ({pareto_size} > {global_best_pareto_size} solutions)")

                    # EĞER KAZANDIYSA GÜNCELLE
                    if is_new_winner:
                        global_best_cost = op_cost
                        global_best_solution = best_sol_of_algo
                        global_best_pareto_size = pareto_size
                        global_best_algo_name = algo_name
                        
                        # EKRANI GÜNCELLE (Hata korumalı)
                        try:
                            if hasattr(self, 'visualizer'):
                                self.root.after(0, lambda s=global_best_solution: self.visualizer.update(s))
                        except:
                            pass

                else:
                    self._log(f"{algo_name} çözüm bulamadı.")
                
            except Exception as e:
                import traceback
                traceback.print_exc()
                self._log(f"ERROR in {algo_name}: {str(e)}")
        
        self.progress_var.set(f"Done! Winner: {global_best_algo_name}")
        self._log("\n" + "="*60)
        self._log(f"Optimization Complete! Winner is: {global_best_algo_name}")
        self._log(f"(Selected because Cost={global_best_cost:.2f} and ParetoSize={global_best_pareto_size})")
        self._log("="*60)

        if len(self.results) > 1:  # Birden fazla algoritma varsa
            self._log("\n" + "="*60)
            self._log("PERFORMANCE METRICS")
            self._log("="*60)
            
            # Metrikleri string olarak al
            import io
            import sys
            old_stdout = sys.stdout
            sys.stdout = buffer = io.StringIO()
            
            comparison.print_comparison()
            
            output = buffer.getvalue()
            sys.stdout = old_stdout
            
            self._log(output)


    def _run_aco_wrapper(self, solver):
        """
        ACO için wrapper (çok sayıda rastgele atama dener).
        Bu fonksiyon DisasterReliefGUI sınıfının içinde olmalıdır.
        """
        from core.solution import ParetoFront
        import numpy as np
        
        front = ParetoFront()
        # ACO tek başına depo seçemediği için ona 10 farklı rastgele senaryo veriyoruz
        for _ in range(10):
            # Rastgele depo ataması yap
            assignments = np.random.randint(0, self.problem.num_depots, self.problem.num_areas)
            # Çöz
            solution = solver.solve(assignments)
            front.add(solution)
            
        return front
    
    def _get_best_solution(self):
        """Helper to get the best solution from PA-LRP or first available"""
        target_algo = 'PA-LRP' if 'PA-LRP' in self.results else list(self.results.keys())[0]
        if target_algo not in self.results or self.results[target_algo]. size() == 0:
            return None, None
        return target_algo, self.results[target_algo]. get_solutions()[0]

    def _show_results(self):
        """Pareto Front Comparison"""
        if not self.results:  return
        Visualizer.plot_pareto_front(self. results).show()

    def _show_route_map_window(self):
        if not self.results: 
            messagebox.showwarning("Warning", "Run optimization first.")
            return

        algo_name, solution = self._get_best_solution()
        if not solution:
            messagebox. showerror("Error", "No valid solution found.")
            return

        fig = Visualizer.plot_dashboard_map(self.problem, solution, title=f"Route Map ({algo_name})")
        plt.show()

    def _show_schedule_window(self):
        if not self.results:
            messagebox.showwarning("Warning", "Run optimization first.")
            return

        algo_name, solution = self._get_best_solution()
        if not solution:
            messagebox.showerror("Error", "No valid solution found.")
            return

        top = tk.Toplevel(self.root)
        top.title(f"Detailed Schedule ({algo_name})")
        top.geometry("600x800")

        text_area = scrolledtext.ScrolledText(top, font=('Consolas', 10))
        text_area.pack(fill=tk.BOTH, expand=True)

        schedule_str = Visualizer.generate_schedule_text(self.problem, solution)
        text_area.insert(tk.END, schedule_str)
        text_area.configure(state='disabled')

    def run(self):
        self.root.mainloop()
        
if __name__ == "__main__": 
    app = DisasterReliefGUI()
    app.run()