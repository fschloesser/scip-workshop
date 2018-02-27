'''
Created on 27.02.2018
'''

from pyscipopt import Model, quicksum
from data.load_cancer import load_cancer
import numpy as np


def initializeWeights(nfeatures):
    """
    Initialize trivial weights that always predict class 1.

    Parameters
    ----------
    nfeatures : int
        positive integer that describes the dimension of the feature space
    """
    weights = [0.0] * nfeatures
    offset = 1.0

def addProblemVariables():
    """
    Add problem variables.
    """
    omegas = []
    # add feature weight variables. add an additional offset
    for f in xrange(nfeatures + 1):
        omegas.append(model.addVar(vtype='C', name="omega_%d" % f, ub=weightBound, lb = -weightBound))

    xis = []
    # add variables xi to penalize misclassification
    for x in xrange(nexamples):
            xis.append(model.addVar(vtype='C', name="psi_%d" %x))

def addObjectiveFunction(X, Y):
    """
    Add the objective function as a quadratic constraint that bounds an artificial objective variable.

    Parameters
    ----------
    X : list
        list of training examples, each example a feature vector list of the same dimension
    Y : list
        labels, either -1 or 1 for each training example
    """
    # add artifical objective function variable
    artobjvar = model.addVar(vtype='C', name="art_obj", obj=1.0)

    # apply correction to account for unequal proportions of positive and negative example
    objCoeffs = [classWeights[0 if yi == -1 else 1] for yi in Y]

    name="objective_function"

    model.addCons(quicksum(.5 *omegas[f]*omegas[f] for f in xrange(nfeatures)) \
                       + quicksum(C / float(nexamples) * objCoeffs[x] * xis[x] for x in xrange(nexamples)) - artobjvar <= 0, name=name)


def addModelFormulation(X, Y, mode='linear'):
    """
    Enrich the model by variables and the necessary constraints

    Parameters
    ----------
    X : list
        training samples with shape (nexamples, nfeatures)
    Y : list
        labels {-1, 1} for every example
    mode : str
        linear or sparse
    """
    # linear
    if mode == "linear":
        for x in xrange(nexamples):
            model.addCons(quicksum(Y[x] * X[x][f] * omegas[f] for f in xrange(nfeatures)) + Y[x] * omegas[nfeatures] >= + 1 - xis[x], name="example_%d"%x)
    # sparse
    elif: mode == "sparse":
        rampLoss = False
        for x in xrange(nexamples):
            model.addCons(quicksum(Y[x] * X[x][f] * omegas[f] for f in xrange(nfeatures)) + Y[x] * omegas[nfeatures] >= + 1 - xis[x], name="example_%d"%x)

        vs = []
        for f in xrange(nfeatures):
            vs.append(model.addVar(vtype='B', name="v_%d" % f))
            model.addCons(omegas[f]  <= weightBound * vs[f], name="upper_vbound_%d"%f)
            model.addCons(omegas[f]  >= -weightBound * vs[f],name="lower_vbound_%d"%f)

        model.addCons(quicksum(vs[f] for f in xrange(nfeatures)) <= int(sparsity * nfeatures), name="sparsity")

def predict_single(X):
    """
    Classify a single example based on its features

    Parameters
    ----------
    X : list or iterable
       feature vector to classify

    Returns
    -------
    y : float
       class prediction for X. Negative means class -1, Positive means class 1
    """
    return offset + sum((weights[f] * X[f] for f in xrange(len(X))))


#if __name__=="__main__":
C = .125
# tLim = 5.0
# verbosity = 0
sparsity = .2
weightBound = 10.0
classWeights= [1.0, 1.0]

# init
# if type(classWeights) is not list or len(classWeights) != 2:
#     return ValueError("'classWeights' argument must be a list of length 2")
#
# if weightBound <= 0 or tLim <= 0 or C <= 0:
#     return ValueError("'C', 'tLim' and 'weightBound' arguments must be positive floats")
#
# if not 0.0 <= sparsity <= 1.0:
#     return ValueError("'sparsity' argument must be between 0.0 and 1.0")

# get data
dataset = load_cancer()
X = np.array(dataset.data)
Y = np.array(dataset.targets)

# fit
model = Model("SCIP-SVM")
# model.setRealParam("limits/time", tLim)
# model.setIntParam("display/verblevel", verbosity)
initializeWeights(len(X[0]))
nexamples = len(Y)

addProblemVariables()
addObjectiveFunction(X, Y)
addModelFormulation(X, Y)

model.optimize()
sols = model.getSols()

if sols:
    weights = [model.getSolVal(sols[0], omegas[f]) for f in xrange(nfeatures)]
    offset = model.getSolVal(sols[0], omegas[nfeatures])

# predict
result = map(predict_single, X)
