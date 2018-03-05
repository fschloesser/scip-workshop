"""
flp.py:  model for solving the capacitated facility location problem

minimize the total (weighted) travel cost from n customers
to some facilities with fixed costs and capacities.

This code represents a modification of an example by
Joao Pedro PEDROSO and Mikio KUBO, 2012

for the SCIP Workshop 2018 at RWTH Aachen, Germany

Author: Gregor Hendel, hendel@zib.de
"""
from pyscipopt import Model, Heur, quicksum, multidict, SCIP_RESULT, SCIP_HEURTIMING

def flp(I, J, d, M, f, c):
    """flp -- model for the capacitated facility location problem
    Parameters:
        - I: set of customers
        - J: set of facilities
        - d[i]: demand for customer i
        - M[j]: capacity of facility j
        - f[j]: fixed cost for using a facility in point j
        - c[i,j]: unit cost of servicing demand point i from facility j
    Returns a model, ready to be solved.
    """


    #
    # model the capacitated facility location by the help of the input data
    #

    model.data = x, y, M, d, c, I


    return model


def make_data():
    I, d = multidict({1:170,
                     2:270,
                     3:250,
                     4:190,
                     5:200,
                     6:230,
                     7:350,
                     8:450,
                     9:200,
                     10:180,
                     11:190
                     }
                    )  # demand
    J, M, f = multidict({1:[450, 1500],
                       2:[500, 800],
                       3:[500, 1200],
                       4:[600, 1900],
                       5:[400, 1000],
                       6:[550, 1900],
                       7:[350, 1800],
                       8:[800, 4000]
                       }
                    )  # capacity, fixed costs
    c = {(1, 1):5, (1, 2):6, (1, 3):9, (1, 4):4, (1, 5):6, (1, 6):4, (1, 7):9, (1, 8):5,  # transportation costs
         (2, 1):5, (2, 2):4, (2, 3):7, (2, 4):3, (2, 5):5, (2, 6):4, (2, 7):5, (2, 8):5,
         (3, 1):6, (3, 2):3, (3, 3):4, (3, 4):3, (3, 5):2, (3, 6):9, (3, 7):6, (3, 8):4,
         (4, 1):8, (4, 2):5, (4, 3):3, (4, 4):4, (4, 5):7, (4, 6):8, (4, 7):8, (4, 8):6,
         (5, 1):10, (5, 2):8, (5, 3):4, (5, 4):5, (5, 5):6, (5, 6):3, (5, 7):6, (5, 8):7,
         (6, 1):11, (6, 2):6, (6, 3):2, (6, 4):3, (6, 5):4, (6, 6):5, (6, 7):9, (6, 8):8,
         (7, 1):11, (7, 2):6, (7, 3):2, (7, 4):3, (7, 5):5, (7, 6):5, (7, 7):2, (7, 8):4,
         (8, 1):11, (8, 2):6, (8, 3):2, (8, 4):5, (8, 5):5, (8, 6):8, (8, 7):3, (8, 8):6,
         (9, 1):4, (9, 2):6, (9, 3):8, (9, 4):9, (9, 5):7, (9, 6):5, (9, 7):8, (9, 8):5,
         (10, 1):3, (10, 2):5, (10, 3):8, (10, 4):7, (10, 5):5, (10, 6):6, (10, 7):7, (10, 8):7,
         (11, 1):6, (11, 2):9, (11, 3):8, (11, 4):3, (11, 5):2, (11, 6):6, (11, 7):6, (11, 8):8
         }
    return I, J, d, M, f, c


class GreedyHeur(Heur):

    def heurexec(self, heurtiming, nodeinfeasible):
        """execution callback of the greedy heuristic

        Parameters:
        -self: the heuristic itself, with the FLP model as attribute
        -heurtiming: timing during the search process
        -nodeinfeasible: flag to indicate whether this node is already infeasible.
        """

        if nodeinfeasible:
            return {"result" : SCIP_DIDNOTRUN}

        sol = self.model.createSol(self)
        x, y, M, d, c, I = self.model.data

        #
        # implement the algorithm of the suggested greedy heuristic
        #
        #

        if False:
            accepted = self.model.trySol(sol)
        else:
            accepted = False

        if accepted:
            return {"result": SCIP_RESULT.FOUNDSOL}
        else:
            return {"result": SCIP_RESULT.DIDNOTFIND}

if __name__ == "__main__":
    I, J, d, M, f, c = make_data()
    model = flp(I, J, d, M, f, c)
    model.includeHeur(GreedyHeur(), "greedyfacility", "greedy heuristic for capacitated facility location", "Y", timingmask = SCIP_HEURTIMING.BEFORENODE)
    model.optimize()

    EPS = 1.e-6
    x, y, M, d, c, I = model.data
    edges = [(i, j) for (i, j) in x if model.getVal(x[i, j]) > EPS]
    facilities = [j for j in y if model.getVal(y[j]) > EPS]

    print("Optimal value:", model.getObjVal())
    print("Facilities at nodes:", facilities)
    print("Edges:", edges)

    #
    # comment this in to get very detailed solving statistics
    #
    #model.printStatistics()

    try:  # plot the result using networkx and matplotlib
        import networkx as NX
        import matplotlib.pyplot as P
        P.clf()
        G = NX.Graph()

        other = [j for j in y if j not in facilities]
        customers = ["c%s" % i for i in d]
        G.add_nodes_from(facilities)
        G.add_nodes_from(other)
        G.add_nodes_from(customers)
        for (i, j) in edges:
            G.add_edge("c%s" % i, j)

        position = NX.drawing.layout.spring_layout(G)
        NX.draw(G, position, node_color = "y", nodelist = facilities)
        NX.draw(G, position, node_color = "g", nodelist = other)
        NX.draw(G, position, node_color = "b", nodelist = customers)
        P.show()
    except ImportError:
        print("install 'networkx' and 'matplotlib' for plotting")
