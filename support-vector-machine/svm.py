
from pyscipopt import Model, quicksum
from data.load_cancer import load_cancer
import numpy as np
import random


#############    Get user input
#############################################################################

mode = "sparse" # mode is in ['linear', 'sparse']

# sparsity of classifier: percentage of present features not being used for classification
sparsity = .5  # between 0.0 and 1.0 (1: all features are used, 0: none are used)

# C is a regularization parameter
C = 100.0 # positive float

# set the bounds for the weights. Weights will be between [-weightBound, weightBound]
weightBound = 1000.0 # positive float
# set these to [1.0, 1.0] if you don't want to manually balance anything
balance = [1.0, 1.0] # list of length 2

# set the time limit
tLim = 100.0 # positive float

# train on how many samples?
n_train = 568

#############    Load data
#############################################################################

# get data
dataset = load_cancer()
X_orig = np.array(dataset.data)
Y_orig = np.array(dataset.targets)

#############    Initialize variables
#############################################################################

nfeatures = len(X_orig[0]) # number of features (dimension of space)
n = len(Y_orig)

randnums = random.sample(range(n), n_train)

# save the first n datapoints for prediction
X = X_orig[randnums]
Y = Y_orig[randnums]

diff = [n for n in range(n) if not n in randnums]
diff = range(n)

X_predict = X_orig[diff]
Y_predict = Y_orig[diff]

nexamples = len(Y) # number of training examples
n = len(Y_predict)

#############    Model
#############################################################################

# create model
model = Model("SCIP-SVM")
# set timelimit
model.setRealParam("limits/time", tLim)

#############    Add problem variables
#############################################################################

omegas = []
# add weight variables and offset variable (as last one)
for f in range(nfeatures + 1):
    omegas.append(model.addVar(vtype='C',
                               name="omega_%d" % f,
                               ub = weightBound,
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

# apply correction to account for unequal proportions of positive and negative examples
balance_coeffs = [balance[0 if yi == -1 else 1] for yi in Y]

model.addCons(quicksum(.5 *omegas[f]*omegas[f] for f in range(nfeatures)) \
                   + quicksum(C / float(nexamples) * balance_coeffs[x] * xis[x] for x in range(nexamples))
                   - objvar <= 0,
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
primalbound = model.getPrimalbound()

#############    Predict
#############################################################################

pairs = np.array(list(zip(X_predict, Y_predict)))
signed_penalties = list(map(
        lambda x:
            (x[1]*max(0, 1 - (offset + sum(weights[f] * x[0][f] for f in range(len(x[0]))))*x[1])),
        pairs))


n_wrong_class = sum([0 if p==0 else 1 for p in signed_penalties])
n_false_positives = sum([1 if p<0 else 0 for p in signed_penalties])
n_false_negatives = sum([1 if p>0 else 0 for p in signed_penalties])

print("")
print("RESULTS:")
print("")
print("mode = {}".format(mode))
print("sparsity = {}".format(sparsity))
print("C = {}".format(C))
print("weightBound = {}".format(weightBound))
print("balance = {}".format(balance))
print("tlim = {}".format(tLim))
print("")
print("Trained on {} datapoints".format(len(Y)))
print("Predicted on {} datapoints".format(len(Y_predict)))
print("")
print("W = {}".format(weights))
print("b = {}".format(offset))
print("obj = {}".format(primalbound))
print("")
print("Number of missclassifications: {} out of {} ({} %)".format(n_wrong_class, n, 100*n_wrong_class/n))
print("Number of false positives: {} out of {} ({} %)".format(n_false_positives, n, 100*n_false_positives/n))
print("Number of false negatives: {} out of {} ({} %)".format(n_false_negatives, n, 100*n_false_negatives/n))

