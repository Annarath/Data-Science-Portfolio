#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 12:44:03 2022

@author: unun
"""
"""
Prblem:

VirtualFood.com is a new company that allows customers to do grocery shopping 
over the Internet. Fresh items, such as fruit, require special handling so they 
always look fresh and do not spoil. Frozen items need cool or frozen storage so 
that they do not warm up. All the other items, including canned goods, paper products, 
and kitchenware, require minimal special attention and are just stocked on shelfs.

The warehouse for storing these items has 85000 square meters of storage space.
The Marketing Department thinks that no more than 50000 square meters should be 
devoted to fresh goods, 50000 to frozen goods, and 60000 to all other items.
The accounting department estimates that the storage cost is £4 per square meter for 
fresh items, £13 per square meter for frozen items, and £0.5 per square meter for all other items. 
Company policy limits the storage cost to £434000.

Frozen goods have the highest profit margin at £65 per square meter, followed by 
fresh items at £30 per square meter, and £6 per square meter from the other items.
What is the maximal profit (profit from goods minus storage costs) that can be achived 
by designing the optimal floorspace allocation?

Model:
    
                  fresh  frozen  other  resource
space                 1       1    1.0     85000
fresh_lim_space       1       0    0.0     50000
frozen_lim_space      0       1    0.0     50000
other_lim_space       0       0    1.0     60000
storage_cost          4      13    0.5    434000
profit_margin        30      65    6.0         0

Objective: max z = (30A + 65B + 6C) - (4A + 13B + 0.5C)

Constraints:
    1A <= 50000
    1B <= 50000
    1C <= 60000
    4A + 13B + 0.5C <= 434000
    1A + 1B + 1C <= 85000
"""

import pandas as pd
import pulp as pl

df = pd.read_excel('2249196_OM_Assignment.xlsx', 'Problem2', index_col=0)

#Objective function:
#Create dictionary of Price_per_unit of each Product
profit = df.loc[df.index[-1], df.columns[0:-1]].to_dict()
cost = df.loc[df.index[-2], df.columns[0:-1]].to_dict()

#Constraints functions:
#Create dataframe for the constraints matrix (exclude first constraints as I will add it later)
constraints_matrix = pd.DataFrame(df, index=df.index[1:-1], columns=df.columns[0:-1]).to_dict('index')

#Create dictionary for the RHS (right hand side) associated with constraints
rhs_coefficient = df.loc[df.index[1:-1], df.columns[-1]].to_dict()

"""
Now we have all the functions to create model to solve assessment Problem 2
"""
#Create a Linear Program Model which has a "Maximisation" Objective
lp_model1 = pl.LpProblem("Grocery_Problem", pl.LpMaximize)

#Define Decision Variables (the variables cannot be lower than 0)
#For (1A, 1B, 1C >= 0)
variables = pl.LpVariable.dicts("space", profit, lowBound=(0))

#add/define the objective function to the model
lp_model1 += pl.lpSum([profit[i]*variables[i] for i in profit]) - pl.lpSum(cost[m]*variables[m] for m in cost)

#add/define the constraints function to the model
for c in rhs_coefficient:
    lp_model1 += pl.lpSum(constraints_matrix[c][u]*variables[u] for u in profit) <= rhs_coefficient[c], c
    
lp_model1 += pl.lpSum(variables) <= 85000

#Solve the model with the default solver
lp_model1.solve()

#Print the status of the solution
print("status:", pl.LpStatus[lp_model1.status])

#Print the obtimised objective function
print("Total Profit (after minus cost) =", pl.value(lp_model1.objective))

#Print the resolved optimum value of each variables
if (pl.LpStatus[lp_model1.status] == "Optimal"):
    for v in lp_model1.variables():
        print(v.name, "=", v.varValue)

#Print Advance info about the constraints for the solution found (do not need this in the assessment)
#see https://realpython.com/linear-programming-python/
print('\n Space Available after allocation')
for name, constraints in lp_model1.constraints.items():
    print(f"{name}: {constraints.value():.2f}")

#Print the model summary to check our objectives, constraints, and variables
print('\n', lp_model1)

#Create file with .lp
lp_model1.writeLP("Grocery_Problem.lp")