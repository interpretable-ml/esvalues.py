import unittest
from esvalues import OrderedSubsets, ESKernelSubsets, esvalues
import numpy as np
import scipy
import itertools
import math
import copy

class TestIterators(unittest.TestCase):

    def subsetsTest(self, members, f, itr, order="descending"):
        alreadySeen = {}
        lastValue = float("inf") if order == "descending" else float("-inf")
        totalCount = 0
        for s in itr:
            if type(s) == tuple and len(s) == 2:
                s = s[0]
            else:
                s = s
            if len(s) > 0:
                if order == "descending":
                    self.assertTrue(f(s) <= lastValue)
                else:
                    self.assertTrue(f(s) >= lastValue)

                lastValue = f(s)

            alreadySeen[tuple(s)] = alreadySeen.get(tuple(s), 0) + 1
            self.assertTrue(alreadySeen[tuple(s)] == 1)
            totalCount += 1

        self.assertTrue(totalCount == 2**len(members))

    def test_minimal(self):
        # test orderedsubsets using a minimum function
        members = np.array([4.19897,0.0835427,0.0452934,1.77052,0.0945746,4.88804,0.135961])
        self.subsetsTest(members, min, OrderedSubsets(members, min))

    def test_reject_large(self):

        # test orderedsubsets using a value that rejects subsets larger than a given value
        P = 7
        def subsetValue(x):
            s = len(x)
            if s == 0:
                return 1e12
            elif s > P/2:
                return -1

            w = (P-1)/(scipy.special.binom(P,s)*s*(P-s))
            return min(x)*w

        members = np.array([4.19897,0.0835427,0.0452934,1.77052,0.0945746,4.88804,0.135961])
        self.subsetsTest(members, subsetValue, OrderedSubsets(members, subsetValue))

    def test_ascending_max(self):
        # test orderedsubsets ascending using a maximum function
        members = np.array([4.19897,0.0835427,0.0452934,1.77052,0.0945746,4.88804,0.135961])
        self.subsetsTest(members, max, OrderedSubsets(members, max, "ascending"), "ascending")

    def test_ascending_sum(self):
        # test orderedsubsets ascending using a sum function
        members = np.array([4.19897,0.0835427,0.0452934,1.77052,0.0945746,4.88804,0.135961])
        self.subsetsTest(members, sum, OrderedSubsets(members, sum, "ascending"), "ascending")

    def test_ESKernelSubsets(self):
        # test eskernelsubsets using a value that rejects subsets larger than a given value
        P = 7
        members = np.arange(0,P,1)
        variances = np.array([4.19897,0.0835427,0.0452934,1.77052,0.0945746,4.88804,0.135961])
        def subsetValue2(x):
            s = x.size
            if s == P or s == 0:
                return 1e12
            elif s > P/2:
                return -1
            elif s == 1:
                return 1e10 + min(variances[x])


            w = (P-1)/(scipy.special.binom(P,s)*s*(P-s))
            min(variances[x])*w

        self.subsetsTest(members, subsetValue2, ESKernelSubsets(members, variances))

class TestESValues(unittest.TestCase):

    def test_minimal(self):
        # basic test
        P = 5
        X = np.zeros((1, P))
        x = np.ones((1, P)) #np.random.randn(1, P)
        f = lambda x: np.sum(x, 1)
        fnull,phi,phiVar = esvalues(x, f, X, nsamples=10)
        self.assertTrue(fnull == 0)
        #print(phi, x, phiVar)
        self.assertTrue(np.linalg.norm(phi - x) < 1e-5)
        self.assertTrue(np.linalg.norm(phiVar) < 1e-5)
        self.assertTrue(phi.size == P)
        self.assertTrue(phiVar.size == P)

    def test_minimal_rand(self):
        # basic test rand
        np.random.seed(1)
        P = 5
        X = np.zeros((1, P))
        x = np.random.randn(1, P)
        f = lambda x: np.sum(x, 1)
        fnull,phi,phiVar = esvalues(x, f, X, nsamples=10)
        self.assertTrue(fnull == 0)
        self.assertTrue(np.linalg.norm(phi - x) < 1e-5)
        self.assertTrue(np.linalg.norm(phiVar) < 1e-5)
        self.assertTrue(phi.size == P)
        self.assertTrue(phiVar.size == P)

    def test_group_no_vary(self):
        P = 5
        # check computation of groups when nothing varies
        X = np.ones((4, P))
        x = np.ones((1, P))
        f = lambda x: np.sum(x, 1)
        groups = [np.array([0,1]),np.array([2]),np.array([3]),np.array([4])]
        fnull,phig,phiVar = esvalues(x, f, X, featureGroups=groups, nsamples=8)
        self.assertTrue(phig.size == 4)

    def test_group(self):
        P = 5
        # check computation of groups when one thing varies
        X = np.ones((4, P))
        x = np.ones((1, P))
        x[0,0] = 0
        f = lambda x: np.sum(x, 1)
        groups = [np.array([0,1]),np.array([2]),np.array([3]),np.array([4])]
        fnull,phig,phiVar = esvalues(x, f, X, featureGroups=groups, nsamples=8)
        self.assertTrue(phig.size == 4)

    def test_many_features(self):
        # check computation with many features
        P = 200
        X = np.zeros((4, P))
        x = np.ones((1, P))
        f = lambda x: np.sum(x, 1)
        fnull,phi,phiVar = esvalues(x, f, X, nsamples=400)
        self.assertTrue(phi.size == 200)

    def test_the_rest(self):
        # make sure things work when only two features vary
        P = 5
        x = np.ones((1, P))
        x[0:2,0] = 0
        X = np.ones((1, P))
        f = lambda x: np.sum(x, 1)
        fnull,phi,phiVar = esvalues(x, f, X, nsamples=8000)
        self.assertTrue(fnull == 5)
        self.assertTrue(np.linalg.norm(X + phi - x) < 1e-5)
        self.assertTrue(np.linalg.norm(phiVar) < 1e-5)

        # X and x are identical
        np.random.seed(1)
        X = np.random.randn(1, P)
        fnull,phi,phiVar = esvalues(x, f, X, nsamples=10)
        self.assertTrue(np.abs(fnull - np.sum(X,1)[0]) < 1e-5)
        self.assertTrue(np.linalg.norm(X + phi - x) < 1e-5)
        self.assertTrue(np.linalg.norm(phiVar) < 1e-5)

        # X and x are identical
        np.random.seed(1)
        x = np.zeros((1, P))
        x[0,0] = 1
        X = np.zeros((1, P))
        fnull,phi,phiVar = esvalues(x, f, X, nsamples=8)
        self.assertTrue(fnull == np.sum(X))
        self.assertTrue(np.linalg.norm(X + phi - x) < 1e-5)
        self.assertTrue(np.linalg.norm(phiVar) < 1e-5)

        # non-zero reference distribution
        X = np.ones((1,P))
        fnull,phi,phiVar = esvalues(x, f, X, nsamples=8000)
        self.assertTrue(fnull == 5)
        self.assertTrue(np.linalg.norm(X + phi - x) < 1e-5)
        self.assertTrue(np.linalg.norm(phiVar) < 1e-5)

        X = np.random.randn(1, P)
        fnull,phi,phiVar = esvalues(x, f, X, nsamples=10)
        self.assertTrue(fnull == f(X)[0])
        self.assertTrue(np.linalg.norm(X + phi - x) < 1e-5)
        self.assertTrue(np.linalg.norm(phiVar) < 1e-5)

        def powerset(iterable):
            "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
            s = list(iterable)
            return itertools.chain.from_iterable(itertools.combinations(s, r) for r in range(len(s)+1))

        def rawShapley(x, f, X, ind, g=lambda x: x, **kwargs):
            featureGroups = kwargs.get("featureGroups", [np.array([i]) for i in range(0,X.shape[1])])

            M = len(featureGroups)
            val = 0.0
            sumw = 0.0
            for s in powerset(list(set(range(0,M)) - set([ind]))):
                S = len(s)
                w = math.factorial(S)*math.factorial(M - S - 1)/math.factorial(M)
                tmp = copy.copy(X)
                for i in range(0,X.shape[0]):
                    for j in s:
                        for k in featureGroups[j]:
                            tmp[i,k] = x[0,k]

                y1 = g(np.mean(f(tmp)))
                for i in range(0, X.shape[0]):
                    for k in featureGroups[ind]:
                        tmp[i,k] = x[0,k]


                y2 = g(np.mean(f(tmp)))
                val += w*(y2-y1)
                sumw += w

            self.assertTrue(abs(sumw - 1.0) < 1e-6)
            return val


        # check brute force computation of groups
        X = np.random.randn(4, P)
        f = lambda x: np.sum(x, 1)
        groups = [np.array([0,1]),np.array([2]),np.array([3]),np.array([4])]
        self.assertTrue(abs(rawShapley(x, f, X, 0) + rawShapley(x, f, X, 1) - rawShapley(x, f, X, 0, featureGroups=groups)) < 1e-5)

        # check computation of groups
        X = np.random.randn(4, P)
        f = lambda x: np.sum(x, 1)
        groups = [np.array([0,1]),np.array([2]),np.array([3]),np.array([4])]
        fnull,phi,phiVar = esvalues(x, f, X, nsamples=10)
        fnull,phig,phiVar = esvalues(x, f, X, featureGroups=groups, nsamples=8)
        self.assertTrue(abs(phi[0] + phi[1] - phig[0]) < 1e-5)

        # check computation of groups when nothing varies
        X = np.ones((4, P))
        x = np.ones((1, P))
        f = lambda x: np.sum(x, 1)
        groups = [np.array([0,1]),np.array([2]),np.array([3]),np.array([4])]
        fnull,phi,phiVar = esvalues(x, f, X, nsamples=10)
        fnull,phig,phiVar = esvalues(x, f, X, featureGroups=groups, nsamples=8)
        self.assertTrue(abs(phi[0] + phi[1] - phig[0]) < 1e-5)

        # check against brute force computation
        X = np.random.randn(4, P)
        f = lambda x: np.sum(x, 1)
        fnull,phi,phiVar = esvalues(x, f, X, nsamples=10)
        for i in range(0, len(phi)):
            self.assertTrue(abs(phi[i] - rawShapley(x, f, X, i)) < 1e-5)

        # non-linear function
        f = lambda x: np.sum(x, 1)**2
        fnull,phi,phiVar = esvalues(x, f, X, nsamples=10)
        for i in range(0, len(phi)):
            self.assertTrue(abs(phi[i] - rawShapley(x, f, X, i)) < 1e-5)

        # non-linear function that interestingly is still possible to estimate with only 2P samples
        f = lambda x: np.sum(x**2, 1)**2
        fnull,phi,phiVar = esvalues(x, f, X, nsamples=10)
        for i in range(0, len(phi)):
            self.assertTrue(abs(phi[i] - rawShapley(x, f, X, i)) < 1e-5)

        def logistic(x):
            return 1/(1+np.exp(-x))
        def logit(x):
            return np.log(x/(1-x))

        # non-linear logistic function
        f = lambda x: logistic(np.sum(x, 1))
        fnull,phi,phiVar = esvalues(x, f, X, nsamples=10000)
        for i in range(0, len(phi)):
            self.assertTrue(abs(phi[i] - rawShapley(x, f, X, i)) < 1e-5)

        # non-linear logistic function with groups
        f = lambda x: logistic(np.sum(x, 1))
        groups = [np.array([0,1]),np.array([2]),np.array([3]),np.array([4])]
        fnull,phi,phiVar = esvalues(x, f, X, featureGroups=groups, nsamples=10000)
        phiRaw = [rawShapley(x, f, X, i, featureGroups=groups) for i in range(0, len(phi))]
        for i in range(0, len(phi)):
            self.assertTrue(abs(phi[i] - phiRaw[i]) < 1e-5)

        # test many totally arbitrary functions
        def gen_model(M):
            model = {}
            for k in powerset(range(0,M)):

                model[tuple(k)] = np.random.randn()

            return model

        P = 10
        X = np.zeros((2, P))
        x = np.ones((1,P))
        for i in range(10):
            model = gen_model(P)
            f = lambda x: np.array([model[tuple(np.nonzero(x[j,:].flatten())[0])] for j in range(x.shape[0])])
            fnull,phi,phiVar = esvalues(x, f, X, nsamples=1000000)
            phiRaw = np.array([rawShapley(x, f, X, j) for j in range(P)])
            self.assertTrue(np.linalg.norm(phi - np.array([rawShapley(x, f, X, j) for j in range(P)])) < 1e-6)

        # non-linear logistic function with logit link
        X = np.random.randn(1,P)
        x = np.random.randn(1,P)
        f = lambda x: logistic(np.sum(x, 1))
        fnull,phi,phiVar = esvalues(x, f, X, logit, nsamples=1000000)
        for i in range(phi.size):
            sv = rawShapley(x, f, X, i, logit)
            self.assertTrue(abs(phi[i] - rawShapley(x, f, X, i, logit)) < 1e-5)

        self.assertTrue(sum(abs(phiVar)) < 1e-12) # we have exhausted the sample space so there should be no uncertainty

        # non-linear logistic function with logit link and random background
        P = 2
        X = np.random.randn(10,P)
        x = np.random.randn(1,P)
        f = lambda x: logistic(np.sum(x, 1))
        fnull,phi,phiVar = esvalues(x, f, X, logit, nsamples=1000000)
        phiRaw = np.array([rawShapley(x, f, X, i, logit) for i in range(phi.size)])
        self.assertTrue(np.linalg.norm(phi - phiRaw) < 1e-5)

        # test many totally arbitrary functions with logit link
        P = 10
        X = np.zeros((2,P))
        x = np.ones((1,P))
        for i in range(10):
            model = gen_model(P)
            f = lambda x: np.array([logistic(model[tuple(np.nonzero(x[j,:].flatten())[0])]) for j in range(x.shape[0])])
            fnull,phi,phiVar = esvalues(x, f, X, logit, nsamples=1000000)
            phiRaw = np.array([rawShapley(x, f, X, j, logit) for j in range(P)])
            self.assertTrue(np.linalg.norm(phi - phiRaw) < 1e-6)

        # test arbitrary functions with logit link and feature groups
        P = 10
        X = np.zeros((2,P))
        x = np.ones((1,P))
        groups = [np.array([0,1]),np.array([2]),np.array([3]),np.array([4,5]),np.array([6,7,8,9])]
        for i in range(3):
            model = gen_model(P)
            f = lambda x: np.array([logistic(model[tuple(np.nonzero(x[j,:].flatten())[0])]) for j in range(x.shape[0])])
            fnull,phi,phiVar = esvalues(x, f, X, logit, nsamples=1000000, featureGroups=groups)
            phiRaw = np.array([rawShapley(x, f, X, j, logit, featureGroups=groups) for j in range(len(groups))])
            self.assertTrue(np.linalg.norm(phi - phiRaw) < 1e-6)
            self.assertTrue(abs(logistic(logit(fnull)+np.sum(phi)) - f(x)[0]) < 1e-6)
