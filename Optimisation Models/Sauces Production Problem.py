#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 16:22:24 2022

@author: unun
"""
"""
              a_c   a_t   a_o   a_l   b_c   b_t   b_o   b_l  resources
profit       2.75  2.75  2.75  2.75  4.00  4.00  4.00  4.00        0.0
cabbage      1.00  0.00  0.00  0.00  1.00  0.00  0.00  0.00      360.0
tomatoes     0.00  1.00  0.00  0.00  0.00  1.00  0.00  0.00      450.0
onions       0.00  0.00  1.00  0.00  0.00  0.00  1.00  0.00      180.0
makeA_c_max  0.25 -0.75 -0.75 -0.75  0.00  0.00  0.00  0.00        0.0
makeB_t_max  0.00  0.00  0.00  0.00 -0.75  0.25 -0.75 -0.75        0.0
makeA_o_max -0.11 -0.11  0.89 -0.11  0.00  0.00  0.00  0.00        0.0
makeB_o_max  0.00  0.00  0.00  0.00 -0.11 -0.11  0.89 -0.11        0.0
makeA_l_max -0.12 -0.12 -0.12  0.88  0.00  0.00  0.00  0.00        0.0
makeB_l_max  0.00  0.00  0.00  0.00 -0.12 -0.12 -0.12  0.88        0.0
sauce_can    1.00  1.00  1.00  1.00  1.00  1.00  1.00  1.00      810.0
makeA_c_min  0.35 -0.65 -0.65 -0.65  0.00  0.00  0.00  0.00        0.0
makeB_t_min  0.00  0.00  0.00  0.00 -0.65  0.35 -0.65 -0.65        0.0
makeA_o_min -0.03 -0.03  0.97 -0.03  0.00  0.00  0.00  0.00        0.0
makeB_o_min  0.00  0.00  0.00  0.00 -0.05 -0.05  0.95 -0.05        0.0
makeA_l_min -0.07 -0.07 -0.07  0.93  0.00  0.00  0.00  0.00        0.0
makeB_l_min  0.00  0.00  0.00  0.00 -0.09 -0.09 -0.09  0.91        0.0
oil          0.00  0.00  0.00  1.00  0.00  0.00  0.00  1.00        0.0
demand       1.00  1.00  1.00  1.00 -1.15 -1.15 -1.15 -1.15        0.0

Objective: max z = 2.75(a_c + a_t + a_o + a_l) + 4(b_c + b_t + b_o + b_l)
    
Constraints:
    1.15(a_c + a_t + a_o + a_l) + b_c + b_t + b_o + b_l <= 810
    
    a_c + b_c                       <= 360
    a_t + b_t                       <= 450
    a_o + b_o                       <= 180
    
    0.25a_c - 0.75(a_t + a_o + a_l) <= 0
    0.25b_t - 0.75(b_t + b_o + b_l) <= 0
    0.97a_o - 0.03(a_c + a_t + a_l) <= 0
    0.95b_o - 0.05(b_c + b_t + b_l) <= 0
    0.93a_l - 0.07(a_c + a_t + a_o) <= 0
    0.91b_l - 0.09(b_c + b_t + b_o) <= 0
    
    0.35a_c - 0.65(a_t + a_o + a_l) >= 0
    0.35b_c - 0.65(b_t + b_o + b_l) >= 0
    0.89a_o - 0.11(a_c + a_t + a_l) >= 0
    0.89b_o - 0.11(b_c + b_t + b_l) >= 0
    0.88a_l - 0.12(a_c + a_t + a_o) >= 0
    0.88a_l - 0.12(b_c + b_t + b_o) >= 0
    
    demand: amount_a_c + amount_a_l + amount_a_o + amount_a_t - 1.15 amount_b_c
 - 1.15 amount_b_l - 1.15 amount_b_o - 1.15 amount_b_t >= 0
 
    oil: amount_a_l + amount_b_l >= 0
"""

import pandas as pd
import pulp as pl

df = pd.read_excel('2249196_OM_Assignment.xlsx', 'Problem5', index_col=0).fillna(0)

#Objective function:
#Create dictionary of profit_per_kg of each sauce
profit_per_kg = df.loc[df.index[0], df.columns[0:-1]].to_dict()
demand = df.loc[df.index[10], df.columns[0:-1]].to_dict()
profit_ratio = df.loc[df.index[0], df.columns[0:-1]]*df.loc[df.index[10], df.columns[0:-1]]
profit_ratio = profit_ratio.to_dict()

#Constraints functions:
#Create dataframe for the constraints matrix
constraints_matrix1 = pd.DataFrame(df, index=df.index[1:11], columns=df.columns[0:-1]).to_dict('index')
constraints_matrix2 = pd.DataFrame(df, index=df.index[11:], columns=df.columns[0:-1]).to_dict('index')

#Create dictionary for the RHS (right hand side) associated with constraints
rhs_coefficient1 = df.loc[df.index[1:11], df.columns[-1]].to_dict()
rhs_coefficient2 = df.loc[df.index[11:], df.columns[-1]].to_dict()

"""
Now we have all the functions to create model to solve the LP Problem 2
"""
#Create a Linear Program Model which has a "Maximisation" Objective
#We can use pulp function "pl.LpProblem()"
lp_model1 = pl.LpProblem("Sauces_Problem", pl.LpMaximize)

#Define Decision Variables (the variables cannot be lower than 0)
#For (1A, 1B, 1C >= 0)
#Create a distionary of variables (continuous variables, by default)
variables = pl.LpVariable.dicts("amount", profit_per_kg, lowBound=(0))

#add/define the objective function to the model
lp_model1 += pl.lpSum([profit_per_kg[i]*variables[i] for i in profit_per_kg])

#add/define the constraints function to the model
for c in rhs_coefficient1:
    lp_model1 += pl.lpSum(constraints_matrix1[c][u]*variables[u] for u in profit_per_kg) <= rhs_coefficient1[c], c

for d in rhs_coefficient2:
    lp_model1 += pl.lpSum(constraints_matrix2[d][e]*variables[e] for e in profit_per_kg) >= rhs_coefficient2[d], d

#Solve the model with the default solver
lp_model1.solve()

#Print the status of the solution
print("status:", pl.LpStatus[lp_model1.status])

#Print the obtimised objective function
print("Total Profit =", pl.value(lp_model1.objective))

#Print the resolved optimum value of each variables
if (pl.LpStatus[lp_model1.status] == "Optimal"):
    for v in lp_model1.variables():
        print(v.name, "=", v.varValue)

#Print Advance info about the constraints for the solution found (do not need this in the assessment)
#see https://realpython.com/linear-programming-python/
print('\n Ingredients left')
for name, constraints in lp_model1.constraints.items():
    print(f"{name}: {constraints.value():.2f}")

#Create file with .lp
lp_model1.writeLP("Sauces_Problem.lp")
