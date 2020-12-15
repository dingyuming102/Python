# Initialization from a BayesianModel object
from pgmpy.factors.discrete import TabularCPD
from pgmpy.models import BayesianModel
from GibbsSamplingWithEvidence import GibbsSampling
import numpy as np
import argparse
import re
import math

# This parses the inputs from test_asia.py (You don't have to modify this!)

parser = argparse.ArgumentParser(description='Asia Bayesian network')
parser.add_argument('--evidence', nargs='+', dest='eVars')
parser.add_argument('--query', nargs='+', dest='qVars')
parser.add_argument('-N', action='store', dest='N')
parser.add_argument('--exact', action="store_true", default=False)
parser.add_argument('--gibbs', action="store_true", default=False)
parser.add_argument('--ent', action="store_true", default=False)

args = parser.parse_args()

print('\n-----------------------------------------------------------')

evidence={}
for item in args.eVars:
    evidence[re.split('=|:', item)[0]]=int(re.split('=|:',item)[1])

print('evidence:', evidence)

query = args.qVars
print('query:', args.qVars)

if args.N is not None:
    N = int(args.N)


# Using TabularCPD, define CPDs

#    +---------+---------+---------+
#    | asia    | asia(0) | asia(1) |
#    +---------+---------+---------+
#    | tub(0)  | 0.99    | 0.95    |
#    +---------+---------+---------+
#    | tub(1)  | 0.01    | 0.05   |
#    +---------+---------+---------+

cpd_tub = TabularCPD(variable='tub', variable_card=2,
                   values=[[0.99, 0.95],
                           [0.01, 0.05]],
                   evidence=['asia'],
                   evidence_card=[2])

cpd_lung = TabularCPD(variable='lung', variable_card=2,
                   values=[[0.99, 0.9],
                           [0.01, 0.1]],
                   evidence=['smoke'],
                   evidence_card=[2])

cpd_bron = TabularCPD(variable='bron', variable_card=2,
                   values=[[0.7, 0.4],
                           [0.3, 0.6]],
                   evidence=['smoke'],
                   evidence_card=[2])

cpd_either = TabularCPD(variable='either', variable_card=2,
                   values=[[0.999, 0.001, 0.001, 0.001],
                           [0.001, 0.999, 0.999, 0.999]],
                  evidence=['lung', 'tub'],
                  evidence_card=[2, 2])

cpd_xray = TabularCPD(variable='xray', variable_card=2,
                   values=[[0.95, 0.02],
                           [0.05, 0.98]],
                   evidence=['either'],
                   evidence_card=[2])

cpd_dysp = TabularCPD(variable='dysp', variable_card=2,
                   values=[[0.9, 0.2, 0.3, 0.1],
                           [0.1, 0.8, 0.7, 0.9]],
                  evidence=['either', 'bron'],
                  evidence_card=[2, 2])


# Defining individual CPDs.
cpd_asia = TabularCPD(variable='asia', variable_card=2, values=[[0.99, 0.01]])
cpd_smoke = TabularCPD(variable='smoke', variable_card=2, values=[[0.5, 0.5]])


# Define edges of the asia model
# Defining the model structure. We can define the network by just passing a list of edges.
model = BayesianModel([('asia', 'tub'), ('smoke', 'lung'), ('smoke', 'bron'), ('tub', 'either'), ('lung', 'either'), ('either', 'xray'), ('either', 'dysp'), ('bron', 'dysp')])


# Associate the parameters with the model structure.
# Associating the CPDs with the network
model.add_cpds(cpd_asia, cpd_smoke, cpd_tub, cpd_lung, cpd_bron, cpd_either, cpd_xray, cpd_dysp)


# check_model checks for the network structure and CPDs and verifies that the CPDs are correctly
# defined and sum to 1.
#model.check_model()


# Find exact solution if args.exact is True:
from pgmpy.inference import VariableElimination
print()
if args.exact is True:
    el = []
    infer = VariableElimination(model)
    for q in query:
        el.append(infer.query([q], evidence=evidence))

    print()
    print("The solutions for exact inference as follow:")
    for e in el:
        print(e)


# Find approximate solution and cross entropy if args.gibbs is True:
import numpy as np
if args.gibbs and args.N is not None:
    # sampling using the provided GibbsSampling class
    gibbs_sampler = GibbsSampling(model)
    samples = gibbs_sampler.sample(evidence=evidence, start_state=None, size=N, return_type="dataframe")
    total_c_e = 0

    print()
    print("The reasonable approximate posterior probabilities as follow:")
    # compute the approximate posterior for each query
    for i, q in enumerate(query):
        qx = sum(samples[q]) / N
        if args.exact:
            # take each p(x)
            px = el[i].values[1]
            # compute the cross-entropy
            total_c_e += - qx * np.log(px) - (1 - qx) * np.log(1 - px)

            # print the reasonable approximate posterior probabilities
            cpd_var = TabularCPD(variable=q, variable_card=2,
                                  values=[[1-qx, qx]])
            print(cpd_var)

# print the cross-entropy
if args.ent:
    print()
    print("The cross-entropy is %.5f" % total_c_e)



print('\n-----------------------------------------------------------')
