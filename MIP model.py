import gurobipy as gp
from gurobipy import GRB
from gurobipy import quicksum
import pandas as pd
import time

start_time = time.time()

# Read facility data from CSV
facility_data = pd.read_csv('facility.csv')
distance_data = pd.read_csv('distance.csv')
vehicle_data = pd.read_csv('vehicle.csv')

# Sets
I = facility_data[facility_data['CATEGORY'] == 'Staging Area']['Facility'].tolist()
J = facility_data[facility_data['CATEGORY'] == 'Evacuating Hospital']['Facility'].tolist()
K = facility_data[facility_data['CATEGORY'] == 'Receiving Hospital']['Facility'].tolist()
pairs = [(j, k) for j in J for k in K]
V = vehicle_data['Vehicle'].tolist()

# Parameters
num_depot = 1  # total number of depots that must be chosen
D = {row['Facility']: row['BEDS'] for _, row in facility_data.iterrows() if row['CATEGORY'] == 'Evacuating Hospital'}
C = {row['Facility']: row['BEDS'] for _, row in facility_data.iterrows() if row['CATEGORY'] == 'Receiving Hospital'}
Time = distance = {(row['From-to'].split(',')[0], row['From-to'].split(',')[1]): row['Distance'] for _, row in distance_data.iterrows()}
M= 10000

# Initialize model
m = gp.Model("MIP_evacuation")

# Decision Variables
u = m.addVars(I+pairs , I+pairs, V, vtype=GRB.BINARY, name="u")
alpha_pv = m.addVars(pairs, V, vtype=GRB.INTEGER, lb=0, name="alpha_pv")
y = m.addVars(I+pairs, V, vtype=GRB.BINARY, name="y")
z_i = m.addVars(I, vtype=GRB.BINARY, name="z_i")

T_max = m.addVar(vtype=GRB.CONTINUOUS, name="T_max")
t_v = m.addVars(V, vtype=GRB.CONTINUOUS, name="t_v")
s = m.addVars(pairs,V, vtype=GRB.CONTINUOUS, name="s") #LB=0


# Objective
#m.setObjective(T_max, GRB.MINIMIZE)
m.setObjective(quicksum(t_v[v] for v in V), GRB.MINIMIZE)

#m.setObjective(
 #   (
  #      quicksum(Time[i, j] * u[i, j, k, v] for i in I for j, k in pairs for v in V) +
   #     quicksum(y[j, k, v] * (2 * alpha_pv[j, k, v] - 1) * Time[j, k] for j, k in pairs for v in V) +
    #    quicksum(Time[k_prime, j] * u[j_prime, k_prime, j, k, v] for j_prime, k_prime in pairs for j, k in pairs if j_prime != j or k_prime != k for v in V) +
     #   quicksum(Time[k, i] * u[j, k, i, v] for i in I for j, k in pairs for v in V)
    #),
    #GRB.MINIMIZE
#)

# Constraints

if len(I)==1:
    m.addConstrs((y[i, v] == 1 for i in I for v in V), name="open_depot1")
    m.addConstrs((z_i[i] == 1 for i in I), name="num_open_depot1")
else:
    m.addConstrs((quicksum(y[i,v] for i in I) == 1 for v in V), name="vehicle_in_depot")  # '*' is used as a wildcard to indicate that we are summing over all indices in the first dimension (in this case, over all depots I)
    m.addConstrs((quicksum(y[i, v] for v in V) <= len(V) * z_i[i] for i in I), name="open_depot")
    m.addConstr((quicksum(z_i[i] for i in I) == num_depot), name="num_open_depot")

#FLOW
m.addConstrs((y[i, v] == quicksum(u[i,j,k,v] for j,k in pairs)  for i in I for v in V), name="depot_to_evacuating")
m.addConstrs((y[i, v] == quicksum(u[j,k,i,v] for j,k in pairs) for i in I for v in V), name="receiving_to_depot")  
m.addConstrs((y[j,k, v] == quicksum(u[j,k,j_prime, k_prime,v] for j_prime, k_prime in pairs if j_prime != j or k_prime != k) + quicksum(u[j,k,i,v] for i in I) for j,k in pairs for v in V), name="outflow")    
m.addConstrs((y[j,k, v] == quicksum(u[j_prime, k_prime,j,k,v] for j_prime, k_prime in pairs if j_prime != j or k_prime != k) + quicksum(u[i,j,k,v] for i in I) for j,k in pairs for v in V), name="inflow")   


#NUMBER OF EVACUEES TO CARRY AND CAPACITIES
m.addConstrs((y[j,k, v] * min(D[j],C[k]) <= alpha_pv[j,k, v] for j,k in pairs for v in V), name="yAlpha")
m.addConstrs((alpha_pv[j,k, v] <= D[j] * y[j,k, v] for j,k in pairs for v in V), name="num_patients_evacuated_jk_by_v")
m.addConstrs((quicksum(alpha_pv[j,k,v] for v in V for j,k in pairs if j==j_prime) == D[j_prime] for j_prime in J), name="demand_must_be_met")
m.addConstrs((quicksum(alpha_pv[j,k,v] for v in V for j,k in pairs if k==k_prime) <= C[k_prime] for k_prime in K),  name="capacity")

#TIME AND SEC CONSTRAINTS (MTZ)
m.addConstrs((s[j,k,v] >= Time[i, j] * u[i, j,k, v] for i in I for j,k in pairs for v in V), name="first_visit_evacuating")
m.addConstrs((s[j,k,v] >= s[j_prime,k_prime,v] + (2 * alpha_pv[j_prime,k_prime, v] - 1) * Time[j_prime,k_prime] + Time[k_prime, j] - M * (1 - u[j_prime,k_prime, j,k, v]) for (j_prime,k_prime) in pairs for (j,k) in pairs if j_prime != j or k_prime != k for v in V), name="pairtopair")
m.addConstrs((t_v[v] >= s[j,k,v] + (2 * alpha_pv[j,k,v] - 1) * Time[j,k] + Time[k, i] - M * (1 - u[j,k,i,v]) for i in I for (j,k) in pairs for v in V), name="returning_to_depot")

#m.addConstrs((T_max >= t_v[v] for v in V), name="maximum_time")

# Optimize model
m.optimize()


import csv
# Check if each solution in the pool is optimal
for i in range(m.SolCount):
    m.setParam(GRB.Param.SolutionNumber, i)  # Select solution i from the pool
    if m.PoolObjVal == m.objVal:
        print('Solution', i+1, 'is optimal with objective value:', m.objVal)
        # Extract the optimal values of decision variables for the current solution
        u_values = m.getAttr('x', u)
        alpha_pv_values = m.getAttr('x', alpha_pv)
        y_values = m.getAttr('x', y)
        z_i_values = m.getAttr('x', z_i)
        s_values = m.getAttr('x', s)
        t_v_values = m.getAttr('x', t_v)

        # Write the optimal values to a CSV file
        with open(f'optimal_values_solution_{i+1}.csv', 'w', newline='') as csvfile:
            fieldnames = ['i', 'j', 'k', 'j_prime', 'k_prime', 'v', 's_jkv', 't_v', 'alpha_pv', 'y_jkv','y_iv', 'u_j_k_j_prime_k_prime_v', 'u_i_j_k_v', 'u_j_prime_k_prime_i_v']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for v in V:
                for i in I:
                    for j, k in pairs:
                        for j_prime, k_prime in pairs:
                            writer.writerow({
                                'i': i,
                                'j': j,
                                'k': k,
                                'j_prime': j_prime,
                                'k_prime': k_prime,
                                'v': v,
                                's_jkv': s_values[j, k, v],
                                't_v': t_v_values[v],
                                'alpha_pv': alpha_pv_values[j, k, v],
                                'y_jkv': y_values[j,k,v],
                                'y_iv': y_values[i,v],
                                'u_j_k_j_prime_k_prime_v': u_values[j, k, j_prime, k_prime, v],
                                'u_i_j_k_v': u_values[i, j, k, v],
                                'u_j_prime_k_prime_i_v': u_values[j_prime, k_prime, i, v]
                            })
                   

print("\nElapsed time:", time.time() - start_time)

