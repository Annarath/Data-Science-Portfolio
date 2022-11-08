#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 27 16:52:23 2022

@author: unun

Problem:

A company must meet the following demands for cash at the beginning of each of the next 
four months: month 1, £120; month 2, £160; month 3, £150; month 4, £130. At the beginning 
of month 1 the company has £299 in cash and £220 worth of bond 1, £250 worth of bond 2, 
and £240 worth of bond 3.
The company have to sell some bonds to meet demands, but the penalty will be charged for 
any bonds sold before the end of month 4. The penalties for selling £1 worth of each bond 
are as shown in the table below. The penalties have to be paid in cash.
What is the minimal cost (i.e. the sum of all penalties paid) of meeting cash demands for 
the next four months? (Fractions of bonds can be sold.)

Model:
    
             month1  month2  month3  month4  month0
cash           1.00    1.00    1.00    1.00   299.0
bond1          0.80    0.60    0.75    0.65   220.0
bond2          0.80    0.75    0.00    0.00   250.0
bond3          0.25    0.85    0.95    0.00   240.0
cash_demand  120.00  160.00  150.00  130.00     0.0

"""

import pandas as pd
import pulp as pl

df = pd.read_excel('2249196_OM_Assignment.xlsx', 'Problem4', index_col=0).fillna(0)

penalty = pd.DataFrame(df, index=df.index[0:-1], columns=df.columns[0:-1]).to_dict('index')

cash_demand = df.loc[df.index[-1], df.columns[0:-1]].to_dict()

assets = df.loc[df.index[0:-1], df.columns[-1]].to_dict()

# Creates the model which is a "Linear Program" with a "Minimisation" 
# objective function
lp_model1 = pl.LpProblem("cash_Bond_Problem", pl.LpMinimize)

# Creates a dictionary of variables. There is a continuous variable 
variables = pl.LpVariable.dicts('cash', (assets, cash_demand), lowBound=0)

# Creates the objective function
lp_model1 += pl.lpSum([penalty[i][j]*variables[i][j] for i in assets for j in cash_demand])

# Adds a constraint to ensure no supply point delivers more than its capacity
for i in assets:
    lp_model1 += pl.lpSum([variables[i][j] for j in cash_demand]) <= assets[i]

# Adds a constraint to ensure each demand point receives exactly as much 
# as its demand
for j in cash_demand:
    lp_model1 += pl.lpSum([variables[i][j] for i in assets]) == cash_demand[j]

# Solves the problem with the default solver
lp_model1.solve()

# The status of the solution is printed to the screen: For an LP, it can be 
# either infeasible, optimal, or unbounded
print("Status:", pl.LpStatus[lp_model1.status])

# The optimised objective function value is printed to the screen if 
# the problem is optimal
print("Total Penalty = ", pl.value(lp_model1.objective), '\n\nThe allocation of cash:')

# Each of the variables is printed with it's resolved optimum value, 
# if the solution is found
if (pl.LpStatus[lp_model1.status] == 'Optimal'):
    for i in assets:
        for j in cash_demand:
            print(variables[i][j].varValue, end='   ')
        print('\n')

print(lp_model1)

# The problem data is written to an .lp file
lp_model1.writeLP("Cash_Bond_Problem.lp")