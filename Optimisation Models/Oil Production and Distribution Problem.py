#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 15:05:40 2022

@author: unun
"""
"""
Problem:

Oilco has oil fields in San Diego and Los Angeles. The San Diego field can produce 
59000 barrels per day, and the Los Angeles field can produce 47000 barrels per day.
Oil is sent from the fields to a refinery, either in Dallas or in Houston (assume that 
each refinery has unlimited capacity). It costs $100 to refine 1000 barrels of oil at Dallas 
and $75 at Houston. Refined oil is shipped to customers in Chicago and New York. Chicago 
customers require 42500 barrels per day of refined oil; New York customers require 42500. 
The costs of shipping 1000 barrels of oil (refined or unrefined) between cities are given 
in the table below.

What is the minimal cost of satisfying demands in refined oil in Chicago and New York? (Assume that it is not allowed to send more oil than required)

Model:
    
                D     H     N     C  supply
la            25.0  30.0   0.0   0.0    47.0
sd            25.0  30.0   0.0   0.0    59.0
d              0.0   0.0  30.0  35.0     0.0
h              0.0   0.0  40.0  25.0     0.0
demand         0.0   0.0  42.5  42.5     0.0
refine_cost  100.0  75.0   0.0   0.0     0.0

Objective: min z = 155laDN + 160laDC + 135laHN + 140laHC + 160sdDN + 150sdDC + 140sdHN + 130sdHC
    
Constraints:
    laDN + laDC + laHN + laHC <= 47
    sdDN + sdDC + sdHN + sdHC <= 59
    laDC + laHC + sdDC + sdHC == 42.5
    laDN + laHN + sdDN + sdHN == 42.5
"""

import pandas as pd
import pulp as pl

df = pd.read_excel('2249196_OM_Assignment.xlsx', 'Problem3', index_col=0).fillna(0)

#Objective function:
#Create dictionary of cost_per_unit of 1k barrel
cost1 = pd.DataFrame(df, index=df.index[0:2], columns=df.columns[0:2]).to_dict('index')
cost2 = pd.DataFrame(df, index=df.index[2:-2], columns=df.columns[2:-1]).to_dict('index')

demand = df.loc[df.index[-2], df.columns[2:-1]].to_dict()

refine_cost = df.loc[df.index[-1], df.columns[0:2]].to_dict()

rhs_supply = df.loc[df.index[0:2], df.columns[-1]].to_dict()

"""
Now we have all the functions to create model to solve assessment Prolem 3
"""
#Create a Linear Program Model which has a "Maximisation" Objective
#We can use pulp function "pl.LpProblem()"
lp_model1 = pl.LpProblem("Oil_Problem", pl.LpMinimize)

#Define Decision Variables (the variables cannot be lower than 0)
#For (1A, 1B, 1C >= 0)
#Create a distionary of variables
variables = pl.LpVariable.dicts('transport', (rhs_supply, cost2, demand), lowBound=0)

#add/define the objective function to the model
lp_model1 += pl.lpSum([(cost1[i][j]+cost2[j][k]+refine_cost[j])*variables[i][j][k] for i in rhs_supply for j in cost2 for k in demand])

#add/define the constraints function to the model
for c in rhs_supply:
    lp_model1 += pl.lpSum(variables[c][u][n] for u in cost2 for n in demand) <= rhs_supply[c], c
    
for n in demand:
    lp_model1 += pl.lpSum(variables[c][u][n] for c in rhs_supply for u in cost2) == demand[n], n

#Solve the model with the default solver
lp_model1.solve()

#Print the status of the solution
print("status:", pl.LpStatus[lp_model1.status])

#Print the obtimised objective function
print("Total Minimised Cost =", pl.value(lp_model1.objective))

#Print the resolved optimum value of each variables
if (pl.LpStatus[lp_model1.status] == "Optimal"):
    for v in lp_model1.variables():
        print(v.name, "=", v.varValue)

#Create file with .lp
lp_model1.writeLP("Oil_Problem.lp")