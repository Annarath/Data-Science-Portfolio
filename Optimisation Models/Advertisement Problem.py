#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 14:14:54 2022

@author: unun
"""
"""
Problem:
    
A local appliance store has decided on an advertising campaign utilising newspaper and radio. 
Each pound spent on newspaper advertising is expected to reach 30 people in 'Under £25,000' and 
30 in the 'Over £25,000' bracket.
Each pound spent on radio advertising is expected to reach 30 people in 'Under £25,000' and 
40 in the 'Over £25,000' bracket.
If the store wants to reach at least 111000 people in the 'Under £25,000' and at least 140000 in 
the 'Over £25,000' bracket, what is the minimum cost of the advertising campaign?

Model:
                newspaper  radio  demand
                                        
under_25               30     30  111000
over_25                30     40  140000
price_per_unit          1      1       0

Objective: min z = 1A + 1B

Constraints:
    30A + 30B >= 111000
    30A + 40B >= 140000
"""

import pulp as pl
import pandas as pd

df = pd.read_excel('2249196_OM_Assignment.xlsx', 'Problem1', index_col=0).fillna(0)

#Objective function:
#Create dictionary of Price_per_unit of each advertising channel
channels = df.loc[df.index[2], df.columns[0:-1]].to_dict()

#Constraints functions:
#Create dataframe for the constraints matrix
constraints_matrix = pd.DataFrame(df, index=df.index[0:-1], columns=df.columns[0:-1]).to_dict('index')

#Create dictionary for the RHS (right hand side) associated with constraints
rhs_coefficient = df.loc[df.index[0:-1], df.columns[-1]].to_dict()

"""
Now we have all the functions to create model to solve the assessment problem 1
"""
#Create a Linear Program Model which has a "Minimisation" Objective
#We can use pulp function "pl.LpProblem()"
lp_model1 = pl.LpProblem("Advertise_Problem", pl.LpMinimize)

#Define Decision Variables (the variables cannot be lower than 0)
#For (1A, 1B >= 0)
#Create a distionary of variables
variables = pl.LpVariable.dicts("invest", channels, lowBound=(0))

#add/define the objective function to the model
lp_model1 += pl.lpSum([channels[i]*variables[i] for i in channels])

#add/define the constraints function to the model
for c in rhs_coefficient:
    lp_model1 += pl.lpSum(constraints_matrix[c][u]*variables[u] for u in channels) >= rhs_coefficient[c], c

#Solve the model with the default solver
lp_model1.solve()

#Print the status of the solution
print("status:", pl.LpStatus[lp_model1.status])

#Print the obtimised objective function
print("Total Minimal Cost =", pl.value(lp_model1.objective))

#Print the resolved optimum value of each variables
if (pl.LpStatus[lp_model1.status] == "Optimal"):
    for v in lp_model1.variables():
        print(v.name, "=", v.varValue)

#Create file with .lp
lp_model1.writeLP("Advertise_Problem.lp")
