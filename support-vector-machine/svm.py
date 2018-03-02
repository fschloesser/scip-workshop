
from pyscipopt import Model, quicksum
from data.load_cancer import load_cancer
import numpy as np


#############    Get user input
#############################################################################

mode = "linear" # mode is in ['linear', 'sparse']

# sparsity of classifier: percentage of present features not being used for classification
sparsity = .2  # between 0.0 and 1.0 (1: all features are used, 0: none are used)

# C is a regularization parameter
C = .125 # positive float

# set the bounds for the weights. Weights will be between [-weightBound, weightBound]
weightBound = 10.0 # positive float
classWeights= [1.0, 1.0] # list of length 2

# set the time limit
tLim = 5.0 # positive float
# set the verbosity level
verbosity = 0 # 0: no verbosity, 1: verbose

#############    Load data
#############################################################################

# get data
dataset = load_cancer()
X = np.array(dataset.data)
Y = np.array(dataset.targets)

#############    Initialize variables
#############################################################################

nfeatures = len(X[0]) # number of features (dimension of space)
nexamples = len(Y) # number of examples

#############    Model
#############################################################################

# create model
model = Model("SCIP-SVM")
# set timelimit
model.setRealParam("limits/time", tLim)
# set verbosity
model.setIntParam("display/verblevel", verbosity)

#############    Add problem variables
#############################################################################

omegas = []
# add feature weight variables
for f in range(nfeatures + 1):
    omegas.append(model.addVar(vtype='C',
                               name="omega_%d" % f,
                               ub=weightBound,
                               lb = -weightBound))

xis = []
# add variables xi to penalize misclassification
for x in range(nexamples):
    xis.append(model.addVar(vtype='C',
                            name="psi_%d" %x))

#############    Add variable for objective function
#############################################################################

# add variable for objective function, since objective function is not linear.
objvar = model.addVar(vtype='C',
                      name="objective",
                      obj=1.0)

# apply correction to account for unequal proportions of positive and negative example
objCoeffs = [classWeights[0 if yi == -1 else 1] for yi in Y]

model.addCons(quicksum(.5 *omegas[f]*omegas[f] for f in range(nfeatures)) \
                   + quicksum(C / float(nexamples) * objCoeffs[x] * xis[x] for x in range(nexamples)) - objvar <= 0,
              name="objective_function")


#############    Add model formulation
#############################################################################

# do this for linear and sparse model
for x in range(nexamples):
    model.addCons(quicksum(Y[x] * X[x][f] * omegas[f] for f in range(nfeatures)) + Y[x] * omegas[nfeatures] >= + 1 - xis[x],
                  name="example_%d"%x)

# sparse
if mode == "sparse":
    vs = []
    for f in range(nfeatures):
        vs.append(model.addVar(vtype='B', name="v_%d" % f))
        model.addCons(omegas[f] <=  weightBound * vs[f],
                      name="upper_vbound_%d"%f)
        model.addCons(omegas[f] >= -weightBound * vs[f],
                      name="lower_vbound_%d"%f)

    model.addCons(quicksum(vs[f] for f in range(nfeatures)) <= int(sparsity * nfeatures),
                  name="sparsity")

#############    Solve / optimize
#############################################################################

model.optimize()

sols = model.getSols()
if not sols:
    print("Could not find solutions, exiting.")
    exit(0)

weights = [model.getSolVal(sols[0], omegas[f]) for f in range(nfeatures)]
offset = model.getSolVal(sols[0], omegas[nfeatures])

#############    Predict
#############################################################################
# predict
result = list(map(
        lambda X:
            offset + sum((weights[f] * X[f] for f in range(len(X)))),
        X))

print(result)