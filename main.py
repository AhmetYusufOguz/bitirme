"""
Ana çalıştırma scripti
Hem komut satırı hem de GUI modları
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt

from core.problem import DisasterReliefProblem
from core.validator import SolutionValidator
from algorithms.pa_lrp import PALRP
from algorithms.aco import ACOSolver
from algorithms.pso import PSOSolver
from algorithms.ap import AP
from metrics.metrics_all import AlgorithmComparison
from gui.gui_complete import DisasterReliefGUI, Visualizer


def run_cli_mode(args):
    """Komut satırı modu"""
    print("\n" + "="*80)
    print("DISASTER RELIEF DISTRIBUTION OPTIMIZER - CLI MODE")
    print("="*80 + "\n")
    
    # Problem oluştur
    print(f"Generating problem instance...")
    print(f"  - Areas: {args.num_areas}")
    print(f"  - Depots: {args.num_depots}")
    print(f"  - Vehicle capacity: {args.vehicle_capacity}")
    
    problem = DisasterReliefProblem.generate_random_instance(
        num_areas=args.num_areas,
        num_depots=args.num_depots,
        seed=args.seed
    )
    problem.vehicle_capacity = args.vehicle_capacity
    
    print("Problem generated successfully!\n")
    
    # Algoritmaları çalıştır
    algorithms_to_run = []
    
    if args.run_pa_lrp:
        algorithms_to_run.append(('PA-LRP', PALRP))
    if args.run_pso:
        algorithms_to_run.append(('PSO', lambda p: PSOSolver(p, ACOSolver)))
    if args.run_aco:
        algorithms_to_run.append(('ACO', ACOSolver))
    if args.run_ap:
        algorithms_to_run.append(('AP', AP))
    
    if not algorithms_to_run:
        print("No algorithms selected! Use --run-pa-lrp, --run-pso, --run-aco, or --run-ap")
        return
    
    results = {}
    comparison = AlgorithmComparison()
    
    for algo_name, AlgoClass in algorithms_to_run:
        print(f"\n{'='*80}")
        print(f"Running {algo_name}...")
        print(f"{'='*80}\n")
        
        if algo_name == 'PA-LRP':
            solver = AlgoClass(
                problem,
                num_particles=args.num_particles,
                num_pso_iterations=args.num_iterations,
                num_ants=args.num_ants,
                num_aco_iterations=20
            )
            pareto_front = solver.solve()
            
        elif algo_name == 'PSO':
            solver = AlgoClass(problem)
            pareto_front = solver.solve()
            
        elif algo_name == 'ACO':
            # ACO için birden fazla rastgele atama dene
            from core.solution import ParetoFront
            solver = AlgoClass(problem, num_ants=args.num_ants, num_iterations=args.num_iterations)
            pareto_front = ParetoFront()
            
            for trial in range(10):
                assignments = np.random.randint(0, problem.num_depots, problem.num_areas)
                solution = solver.solve(assignments)
                pareto_front.add(solution)
                print(f"  Trial {trial+1}/10: f1={solution.f1_penalty_cost:.2f}, f2={solution.f2_operational_cost:.2f}")
        
        elif algo_name == 'AP':
            solver = AlgoClass(problem, num_iterations=args.num_iterations)
            pareto_front = solver.solve()
        
        results[algo_name] = pareto_front
        comparison.add_result(algo_name, pareto_front)
        
        print(f"\n{algo_name} completed!")
        print(f"Pareto front size: {pareto_front.size()}")
    
    # Sonuçları göster
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80 + "\n")
    
    comparison.print_comparison()
    
    # Validator ile en iyi çözümü kontrol et
    if 'PA-LRP' in results and results['PA-LRP'].size() > 0:
        print("\n" + "="*80)
        print("BEST SOLUTION VALIDATION (PA-LRP)")
        print("="*80 + "\n")
        
        best_solution = results['PA-LRP'].get_solutions()[0]
        validator = SolutionValidator(problem)
        stats = validator.get_solution_statistics(best_solution)
        
        print(f"Valid: {stats['is_valid']}")
        print(f"Opened depots: {stats['num_opened_depots']}")
        print(f"Number of routes: {stats['num_routes']}")
        print(f"Average vehicle utilization: {stats['avg_vehicle_utilization']*100:.2f}%")
        print(f"\nObjective values:")
        print(f"  f1 (Penalty): {stats['f1_penalty_cost']:.2f}")
        print(f"  f2 (Operational): {stats['f2_operational_cost']:.2f}")
        print(f"    - Depot opening: {stats['depot_opening_cost']:.2f}")
        print(f"    - Vehicle fixed: {stats['vehicle_fixed_cost']:.2f}")
        print(f"    - Transport: {stats['transport_cost']:.2f}")
        
        if stats['violations']:
            print(f"\nViolations:")
            for v in stats['violations']:
                print(f"  - {v}")
    
    # Grafikleri göster
    if args.show_plots:
        print("\n" + "="*80)
        print("GENERATING PLOTS...")
        print("="*80 + "\n")
        
        # Pareto front karşılaştırması
        Visualizer.plot_pareto_front(results, "Algorithm Comparison - Pareto Fronts")
        plt.savefig('pareto_comparison.png', dpi=150, bbox_inches='tight')
        print("Saved: pareto_comparison.png")
        
        # En iyi çözümün haritası
        if 'PA-LRP' in results and results['PA-LRP'].size() > 0:
            best_solution = results['PA-LRP'].get_solutions()[0]
            Visualizer.plot_routes_on_map(problem, best_solution, "PA-LRP - Best Solution Routes")
            plt.savefig('best_solution_routes.png', dpi=150, bbox_inches='tight')
            print("Saved: best_solution_routes.png")
        
        # Yakınsama grafiği (PA-LRP için)
        if 'PA-LRP' in results and hasattr(solver, 'get_convergence_data'):
            convergence = solver.get_convergence_data()
            Visualizer.plot_convergence(convergence, "PA-LRP Convergence History")
            plt.savefig('convergence.png', dpi=150, bbox_inches='tight')
            print("Saved: convergence.png")
        
        if not args.no_display:
            plt.show()
    
    print("\n" + "="*80)
    print("OPTIMIZATION COMPLETE!")
    print("="*80 + "\n")


def run_gui_mode():
    """GUI modu"""
    print("Starting GUI mode...")
    app = DisasterReliefGUI()
    app.run()


def main():
    parser = argparse.ArgumentParser(
        description='Disaster Relief Distribution Optimizer',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # GUI mode (default)
  python main.py
  
  # CLI mode with PA-LRP only
  python main.py --cli --run-pa-lrp
  
  # CLI mode with all algorithms
  python main.py --cli --run-pa-lrp --run-pso --run-aco --run-ap
  
  # Custom problem size
  python main.py --cli --run-pa-lrp --num-areas 100 --num-depots 10
  
  # Save plots without displaying
  python main.py --cli --run-pa-lrp --show-plots --no-display
        """
    )
    
    # Mode
    parser.add_argument('--cli', action='store_true',
                       help='Run in CLI mode (default: GUI)')
    
    # Problem parameters
    parser.add_argument('--num-areas', type=int, default=50,
                       help='Number of affected areas (default: 50)')
    parser.add_argument('--num-depots', type=int, default=5,
                       help='Number of candidate depots (default: 5)')
    parser.add_argument('--vehicle-capacity', type=float, default=200,
                       help='Vehicle capacity (default: 200)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    
    # Algorithm selection
    parser.add_argument('--run-pa-lrp', action='store_true',
                       help='Run PA-LRP algorithm')
    parser.add_argument('--run-pso', action='store_true',
                       help='Run PSO algorithm')
    parser.add_argument('--run-aco', action='store_true',
                       help='Run ACO algorithm')
    parser.add_argument('--run-ap', action='store_true',
                       help='Run AP algorithm')
    
    # Algorithm parameters
    parser.add_argument('--num-particles', type=int, default=30,
                       help='Number of particles for PSO (default: 30)')
    parser.add_argument('--num-ants', type=int, default=30,
                       help='Number of ants for ACO (default: 30)')
    parser.add_argument('--num-iterations', type=int, default=50,
                       help='Number of iterations (default: 50)')
    
    # Output
    parser.add_argument('--show-plots', action='store_true',
                       help='Generate and save plots')
    parser.add_argument('--no-display', action='store_true',
                       help='Do not display plots (only save)')
    
    args = parser.parse_args()
    
    if args.cli:
        run_cli_mode(args)
    else:
        run_gui_mode()


if __name__ == "__main__":
    main()