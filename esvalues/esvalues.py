from .iterators import ESKernelSubsets

def varying_groups(x, X, featureGroups):
    varying = np.zeros(len(featureGroups))
    for i in range(0,len(featureGroups)):
        inds = featureGroups[i]
        varying[i] = sum(sum(x[0,inds] == X[:,inds]) != inds.size)
    return np.nonzero(varying)[0]


class ESValuesEstimator:
    def __init__(self, f, X, link=lambda x: x, **kwargs):
        N,P = X.shape

        # give default values to omitted arguments
        weights = kwargs.get("weights", np.ones(N))
        weights /= sum(weights)
        featureGroups = kwargs.get("featureGroups", [np.array([i]) for i in range(0,P)])
        assert len(weights) == N,  "Provided 'weights' must match the number of representative data points %(N)!" % locals()

        self.x = np.ones(1)
        self.f = f
        self.X = X
        self.link = link
        self.linkv = np.vectorize(link)
        self.featureGroups = featureGroups
        self.weights = weights
        self.nsamples = kwargs.get("nsamples", 0)
        self.varyingInds = []
        self.varyingFeatureGroups = []
        self.P = P
        self.N = N
        self.M = 0
        self.nsamplesAdded = 0
        self.nsamplesRun = 0

    def allocate(self):
        self.data = np.zeros((self.nsamples * self.N, self.P))
        self.maskMatrix = np.zeros((self.nsamples, self.M-1))
        self.kernelWeights = np.zeros(self.nsamples)
        self.y = np.zeros(self.nsamples * self.N)
        self.ey = np.zeros(self.nsamples)
        self.lastMask = np.zeros(self.nsamples)

    def esvalues(self, x):
        assert x.size == self.P, "Provided 'x' length must match the data matrix features count ("+str(x.size)+" != "+str(self.P)+")!"
        self.x = x

        # find the feature groups we will test. If a feature does not change from its
        # current value then we know it doesn't impact the model
        self.varyingInds = varying_groups(self.x, self.X, self.featureGroups)
        self.varyingFeatureGroups = [self.featureGroups[i] for i in self.varyingInds]
        self.M = len(self.varyingFeatureGroups)

        # find f(x) and E_x[f(x)]
        self.fx = self.f(x)[0]
        self.fnull = np.mean(self.f(self.X))

        # if no features vary then there no feature has an effect
        if self.M == 0:
            return self.fx,np.zeros(self.featureGroups.size),np.zeros(self.featureGroups.size)

        # if only one feature varies then it has all the effect
        elif self.M == 1:
            fx = np.mean(self.f(x))
            fnull = np.mean(self.f(self.X))
            φ = np.zeros(len(self.featureGroups))
            φ[self.varyingInds[0]] = self.link(self.fx) - self.link(self.fnull)
            return self.fnull,φ,np.zeros(len(self.featureGroups))

        # pick a reasonable number of samples if the user didn't specify how many they wanted
        if self.nsamples == 0:
            self.nsamples = 2*self.M+1000

        if self.M <= 30 and self.nsamples > 2**self.M-2:
            self.nsamples = 2**self.M-2

        assert self.nsamples >= min(2*self.M, 2**self.M-2), "'nsamples' must be at least 2 times the number of varying feature groups!"

        # add the singleton samples
        self.allocate()
        itr = ESKernelSubsets(np.arange(0, self.M), np.ones(self.M))
        itr.__next__()
        itr.__next__()
        for i in range(0,min(2*self.M, 2**self.M-2)):
            m,w = itr.__next__()
            self.addsample(x, m, w)
        self.run()

        # if there might be more samples then enumarate them
        if self.y.size >= 2*self.M:

            # estimate the variance of each ES value estimate
            variances = np.zeros(self.M)
            for i in range(0,min(2*self.M, 2**self.M-2),2):
                variances[(i+1)//2] = np.var(np.array([self.y[i] - self.fnull, self.fx - self.y[i+1]]))

            # now add the rest of the samples giving priority to ES values with high estimated variance
            itr = ESKernelSubsets(np.arange(0, self.M), variances)
            for i in range(0,min(2*self.M, 2**self.M-2)+2):
                itr.__next__()
            for i in range(0,self.nsamples-(min(2*self.M, 2**self.M-2))):
                m,w = itr.__next__()
                self.addsample(x, m, w)
            self.run()


        # solve then expand the ES values vector to contain the non-varying features as well
        vφ,vφVar = self.solve()
        φ = np.zeros(len(self.featureGroups))
        φ[self.varyingInds] = vφ
        φVar = np.zeros(len(self.featureGroups))
        φVar[self.varyingInds] = vφVar

        # return the Shapley values along with variances of the estimates
        return self.fnull,φ,φVar

    def addsample(self, x, m, w):
        offset = self.nsamplesAdded * self.N
        for i in range(0,self.N):
            for j in range(0,self.M):
                for k in self.varyingFeatureGroups[j]:
                    if m[j] == 1.0:
                        self.data[offset+i,k] = x[0,k]
                    else:
                        self.data[offset+i,k] = self.X[i,k]

        self.maskMatrix[self.nsamplesAdded,:] = m[0:-1] - m[-1]
        self.lastMask[self.nsamplesAdded] = m[-1]
        self.kernelWeights[self.nsamplesAdded] = w
        self.nsamplesAdded += 1

    def run(self):
        modelOut = self.f(self.data[self.nsamplesRun*self.N:self.nsamplesAdded*self.N,:])
        self.y[self.nsamplesRun*self.N:self.nsamplesAdded*self.N] = modelOut

        # find the expected value of each output
        for i in range(self.nsamplesRun, self.nsamplesAdded):
            eyVal = 0.0
            for j in range(0, self.N):
                eyVal += self.y[i*self.N + j]

            self.ey[i] = eyVal/self.N
            self.nsamplesRun += 1

    def solve(self):
        # adjust the y value according to the constraints for the offset and sum
        eyAdj = self.linkv(self.ey) - self.lastMask*(self.link(self.fx) - self.link(self.fnull)) - self.link(self.fnull)

        # solve a weighted least squares equation to estimate φ
        tmp = np.transpose(np.transpose(self.maskMatrix) * np.transpose(self.kernelWeights))
        tmp2 = np.linalg.inv(np.dot(np.transpose(tmp),self.maskMatrix))
        w = np.dot(tmp2,np.dot(np.transpose(tmp),eyAdj))
        wlast = (self.link(self.fx) - self.link(self.fnull)) - sum(w)
        φ = np.hstack((w, wlast))

        yHat = np.dot(self.maskMatrix, w)
        φVar = np.var(yHat - eyAdj) * np.diag(tmp2)
        φVar = np.hstack((φVar, max(φVar))) # since the last weight is inferred we use a pessimistic guess of its variance

        # a finite sample adjustment based on how much of the weight is left in the sample space
        fractionWeightLeft = 1 - sum(self.kernelWeights)/sum(np.array([(self.M-1)/(s*(self.M-s)) for s in range(1, self.M)]))

        return φ,φVar*fractionWeightLeft

def esvalues(x, f, X, link=lambda x: x, **kwargs):
    return ESValuesEstimator(f, X, link, **kwargs).esvalues(x)
