import matplotlib.pyplot as plt
import numpy as np
from pyscipopt import Model, sqrt, SCIP_PARAMSETTING

#
# Actual solution
#

r = 0.572917
t = np.arange(0.0, 2.42, 0.01)
x = r*(t - np.sin(t))
y = -r*(1 - np.cos(t))

plt.plot(x,y)
#plt.show()


#
# SCIP model
#

N = 4 # number of points is N+1

g = 9.8 # gravity constant

scip = Model()

#
# Create variables
#

x = []
y = []
for i in xrange(N+1):
    x.append(scip.addVar("x%d"%i, ub=1.0))
    y.append(scip.addVar("y%d"%i, lb=-1.0, ub = -0.00001)) # the curve, in principle, could go further down than -1

t = scip.addVar("time", obj = 1.0) # time variable: objective coefficient 1.0

#
# initial and final condition
#
scip.chgVarUb(y[0],0.0)
scip.addCons(y[0] == 0.0)
scip.addCons(x[0] == 0.0)
scip.addCons(y[N] == -1.0)
scip.addCons(x[N] == 1.0)

#
# Optional constraint: y_i are decreasing, x_i are increasing
#
for i in range(N):
    scip.addCons(y[i+1] <= y[i])
    scip.addCons(x[i] <= x[i+1])

#
# discretization expression and time constraint
#
disc = sqrt(2/g) * sum(sqrt(( (x[i+1] - x[i])**2 + (y[i+1] - y[i])**2 )) / ((sqrt(-y[i+1]) + sqrt(-y[i])) ) for i in range(N))
scip.addCons(disc <= t)

#scip.writeProblem()
#scip.setPresolve(SCIP_PARAMSETTING.OFF)

#
# Solve problem
#
scip.optimize()

#
# Get values for plotting
#
for i in range(N+1):
    print(i, scip.getVal(x[i]), scip.getVal(y[i]))

vals = sorted( zip(map(scip.getVal, x), map(scip.getVal, y)), key = lambda x : x[0])
print(vals)
unzipped = zip(*vals)
valsx, valsy = unzipped[0], unzipped[1]

#
# plot and show both plots: scip's solution and analytical
#
plt.plot( np.array(valsx), np.array(valsy))
plt.show()
