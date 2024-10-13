from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import gurobipy as gp
from gurobipy import GRB
import os
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Helper function to optimize routes
def optimize_routes(distance_matrix, demand_array, vehicle_capacity=90):
    num_places = len(demand_array)

    # Initial routes for the master problem (one vehicle to each place)
    initial_routes = []
    initial_route_costs = []

    for i in range(1, num_places):
        initial_routes.append([(0, i), (i, 0)])  # Route: Depot -> Place -> Depot
        initial_route_costs.append(distance_matrix[0, i] + distance_matrix[i, 0])

    # Master problem and subproblem solving logic (using Gurobi)
    def solve_master_problem(routes, route_costs, demands, is_integer=False):
        model = gp.Model("MasterProblem")
        vtype = GRB.BINARY if is_integer else GRB.CONTINUOUS
        x = model.addVars(len(routes), vtype=vtype, name="x")
        model.setObjective(gp.quicksum(route_costs[r] * x[r] for r in range(len(routes))), GRB.MINIMIZE)

        for i in range(1, num_places):
            model.addConstr(gp.quicksum(x[r] for r in range(len(routes)) if i in [j for _, j in routes[r]]) >= 1,
                            f"demand_fulfillment_{i}")
        model.optimize()

        if model.status == GRB.OPTIMAL:
            return model.objVal, model.getAttr("x"), [c.Pi for c in model.getConstrs()] if not is_integer else None
        else:
            raise Exception("Master problem did not converge")

    def solve_subproblem(dual_values):
        model = gp.Model("SubProblem")
        y = model.addVars(num_places, num_places, vtype=GRB.BINARY, name="y")
        z = model.addVars(num_places, vtype=GRB.CONTINUOUS, name="z")
        reduced_cost = gp.quicksum((distance_matrix[i, j] - (dual_values[j - 1] if j > 0 else 0)) * y[i, j]
                                   for i in range(num_places) for j in range(num_places))
        model.setObjective(reduced_cost, GRB.MINIMIZE)

        for i in range(1, num_places):
            model.addConstr(gp.quicksum(y[i, j] for j in range(num_places)) == gp.quicksum(y[j, i] for j in range(num_places)))

        model.addConstr(gp.quicksum(y[0, j] for j in range(1, num_places)) == 1, "start_at_depot")
        model.addConstr(gp.quicksum(y[i, 0] for i in range(1, num_places)) == 1, "end_at_depot")

        for i in range(1, num_places):
            for j in range(1, num_places):
                model.addConstr(z[i] >= z[j] + demand_array[j] - vehicle_capacity * (1 - y[i, j]))

        model.optimize()
        if model.status == GRB.OPTIMAL and model.objVal < -1e-6:
            new_route = []
            current_node = 0
            while True:
                for j in range(num_places):
                    if y[current_node, j].X > 0.5:
                        new_route.append((current_node, j))
                        current_node = j
                        break
                if current_node == 0:
                    break
            return new_route, sum(distance_matrix[i, j] for i, j in new_route)
        else:
            return None, None

    routes = initial_routes[:]
    route_costs = initial_route_costs[:]

    while True:
        _, route_usage, dual_values = solve_master_problem(routes, route_costs, demand_array, is_integer=False)
        new_route, new_route_cost = solve_subproblem(dual_values)

        if new_route is None:
            break
        else:
            routes.append(new_route)
            route_costs.append(new_route_cost)

    _, final_route_usage, _ = solve_master_problem(routes, route_costs, demand_array, is_integer=True)
    selected_routes = [routes[r] for r, usage in enumerate(final_route_usage) if usage > 0.99]
    return selected_routes

# Route to upload files and display form
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        distance_file = request.files['distance_matrix']
        demand_file = request.files['demand_matrix']

        if distance_file and demand_file:
            distance_path = os.path.join(app.config['UPLOAD_FOLDER'], distance_file.filename)
            demand_path = os.path.join(app.config['UPLOAD_FOLDER'], demand_file.filename)
            distance_file.save(distance_path)
            demand_file.save(demand_path)

            # Read Excel files
            distance_df = pd.read_excel(distance_path, index_col=0)
            demand_df = pd.read_excel(demand_path)

            # Extract matrix and demands
            distance_matrix = distance_df.to_numpy()
            demand_array = demand_df.iloc[0, 1:].to_numpy()

            # Optimize routes
            optimized_routes = optimize_routes(distance_matrix, demand_array)

            # Convert routes to URL-friendly format
            optimized_routes_str = [str(route) for route in optimized_routes]

            # Redirect to route display
            return redirect(url_for('display_routes', routes=optimized_routes_str))

    return render_template('upload.html')

# Route to display optimized routes
@app.route('/routes')
def display_routes():
    routes_str = request.args.getlist('routes')
    # Parse the string format back to lists of tuples
    routes = [eval(route) for route in routes_str]
    return render_template('routes.html', routes=routes)

@app.route('/visualize')
def visualize_routes():
    routes_str = request.args.getlist('routes')
    routes = [eval(route) for route in routes_str]

    # Generate graph
    G = nx.Graph()
    for route in routes:
        for edge in route:
            G.add_edge(edge[0], edge[1])

    pos = nx.spring_layout(G)
    plt.figure()
    nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='black', width=2, node_size=700)
    
    # Save the graph to the 'static' folder
    static_folder = os.path.join(app.root_path, 'static')
    if not os.path.exists(static_folder):
        os.makedirs(static_folder)
        
    image_path = 'graph.png'
    full_image_path = os.path.join(static_folder, image_path)
    plt.savefig(full_image_path)
    plt.close()

    # Render the visualize page and pass the image path
    return render_template('visualize.html', image_path=image_path)


# # Visualization route
# @app.route('/visualize')
# def visualize_routes():
#     routes_str = request.args.getlist('routes')
#     routes = [eval(route) for route in routes_str]

#     # Generate graph
#     G = nx.Graph()
#     for route in routes:
#         for edge in route:
#             G.add_edge(edge[0], edge[1])

#     pos = nx.spring_layout(G)
#     plt.figure()
#     nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='black', width=2, node_size=700)
#     image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'graph.png')
#     plt.savefig(image_path)
#     plt.close()

#     return render_template('visualize.html', image_path=image_path)

if __name__ == '__main__':
    app.run()
