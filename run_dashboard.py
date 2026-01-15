import numpy as np
import matplotlib.pyplot as plt
import types

# --- IMPORT PARTNER'S MODULES ---
from core.problem import DisasterReliefProblem
from algorithms.aco import ACO, ACOSolver
from core.solution import Solution, Route

# ==========================================
# ==========================================
def patched_construct_routes(self, depot_id: int, assigned_areas: list) -> list:
    routes = []
    unvisited = set(assigned_areas)
    vehicle_id = 0
    depot_node = depot_id
    temp_solution = Solution(self.problem)

    while unvisited:
        route = Route(depot_id=depot_id, vehicle_id=vehicle_id)
        current_node = depot_node
        current_load = 0.0
        current_time = 0.0
        
        while unvisited:
            next_area = self._select_next_area(current_node, unvisited, current_load, current_time)
            if next_area is None: break
            
            area = self.problem.areas[next_area]
            route.add_area(next_area)
            unvisited.remove(next_area)
            
            current_load += area.demand
            area_node = self.problem.num_depots + next_area
            travel_time = self.problem.get_travel_time(current_node, area_node)
            arrival_time = current_time + travel_time
            service_start = max(arrival_time, area.soft_lower)
            current_time = service_start + area.service_time
            current_node = area_node
        
        if not route.is_empty():
            route = temp_solution.calculate_route_metrics(route)
            routes.append(route)
            vehicle_id += 1
            
    return routes

ACO._construct_routes = patched_construct_routes

# ==========================================
# PART 2: THE BRIDGE (Logic)
# ==========================================
def solve_using_partners_code():
    print("1. Generating Problem (50 Areas, 5 Depots)...")
    problem = DisasterReliefProblem.generate_random_instance(num_areas=50, num_depots=5, map_size=100, seed=42)
    problem.vehicle_capacity = 200.0
    for depot in problem.depots: depot.capacity = 300.0 

    print("2. Assigning Nearest Depots...")
    assignments = np.zeros(problem.num_areas, dtype=int)
    for area in problem.areas:
        dists = [problem.get_distance(d.id, problem.num_depots + area.id) for d in problem.depots]
        assignments[area.id] = np.argmin(dists)

    print("3. Running ACO Solver...")
    aco_solver = ACOSolver(problem, num_ants=20, num_iterations=20)
    solution = aco_solver.solve(assignments)
    return problem, solution

# ==========================================
# PART 3: SEPARATE WINDOWS VISUALIZATION
# ==========================================
def create_separated_dashboard(problem, solution):
    print("4. Rendering Separate Windows...")
    
    depot_locs = np.array([d.location for d in problem.depots])
    area_locs = np.array([a.location for a in problem.areas])
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

    # --- FIGURE 1: THE MAP (SQUARE) ---
    fig1, ax_map = plt.subplots(figsize=(10, 10)) # Square Figure
    
    if solution.routes:
        for i, route in enumerate(solution.routes):
            if route.is_empty(): continue
            path_x = [depot_locs[route.depot_id][0]] + \
                     [area_locs[aid][0] for aid in route.sequence] + \
                     [depot_locs[route.depot_id][0]]
            path_y = [depot_locs[route.depot_id][1]] + \
                     [area_locs[aid][1] for aid in route.sequence] + \
                     [depot_locs[route.depot_id][1]]
            ax_map.plot(path_x, path_y, c=colors[i % len(colors)], linewidth=2, alpha=0.8)

    ax_map.scatter(area_locs[:, 0], area_locs[:, 1], c='blue', s=80, zorder=3, edgecolors='white')
    opened_ids = set(solution.opened_depots)
    
    for i, depot in enumerate(problem.depots):
        color = 'red' if i in opened_ids else 'lightgray'
        size = 200 if i in opened_ids else 50
        marker = 's' if i in opened_ids else 'x'
        ax_map.scatter(depot.location[0], depot.location[1], c=color, marker=marker, s=size, zorder=4, edgecolors='black')
        ax_map.text(depot.location[0], depot.location[1]+3, f'D{i}', fontsize=11, ha='center', fontweight='bold', color='darkred')

    # THIS FIXES THE STRETCHING:
    ax_map.set_aspect('equal') 
    
    ax_map.set_title("Optimized Relief Routes (Map View)", fontsize=16)
    ax_map.grid(True, linestyle='--', alpha=0.4)
    ax_map.set_xlim(-5, 105) # Add padding
    ax_map.set_ylim(-5, 105)

    # --- FIGURE 2: THE SCHEDULE (TEXT) ---
    fig2, ax_text = plt.subplots(figsize=(8, 10)) # Portrait Figure
    ax_text.axis('off')
    
    log_text = "OPTIMIZED DELIVERY SCHEDULE\n"
    log_text += "="*45 + "\n"
    
    if not solution.routes:
        log_text += "\nNo feasible solution found."
    else:
        sorted_routes = sorted(solution.routes, key=lambda r: r.depot_id)
        count = 0
        for i, route in enumerate(sorted_routes):
            if route.is_empty() or count > 20: continue 
            count += 1
            
            log_text += f"\nVEHICLE {i+1} [Depot {route.depot_id}]\n"
            log_text += f"  Stats: {route.total_distance:.1f}km | Load: {int(route.total_demand)}/{int(problem.vehicle_capacity)}\n"
            
            stops = route.sequence
            display_stops = stops[:5] # Show 5 stops max
            
            for area_id in display_stops:
                area = problem.areas[area_id]
                window_str = f"[{int(area.soft_lower)}-{int(area.soft_upper)}]"
                log_text += f"   -> Area {area_id:<2} Window: {window_str}\n"
            
            if len(stops) > 5:
                log_text += "   -> ... (more stops)\n"

        log_text += "\n" + "="*45 + "\n"
        log_text += f"Active Vehicles:  {len(solution.routes)}\n"
        log_text += f"Total Distance:   {solution.transport_cost / problem.transport_cost_rate:.1f} km\n"
        log_text += f"Operational Cost: {solution.f2_operational_cost:.2f}"

    ax_text.text(0.05, 0.98, log_text, transform=ax_text.transAxes, 
                 fontsize=11, family='monospace', verticalalignment='top')

    plt.show()

if __name__ == "__main__":
    try:
        prob, sol = solve_using_partners_code()
        create_separated_dashboard(prob, sol)
    except Exception as e:
        import traceback
        traceback.print_exc()