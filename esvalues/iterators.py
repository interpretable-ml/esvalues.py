
class OrderedSubsets:
    def __init__(self, members, f, order="decending"):
        orderBy = lambda x: f(np.array([x]))
        f2 = f
        if order == "decending": # vs ascending
            orderBy = lambda x: -f(np.array([x]))
            f2 = lambda x: -f(x)
        self.f = f2
        orderBy = np.vectorize(orderBy)
        #print("members", members, type(members))
        #print("self.vf(members)",orderBy(members))
        #print("np.argsort(orderBy(members))", np.argsort(orderBy(members)), type(np.argsort(orderBy(members))))
        self.members = members[np.argsort(orderBy(members))]
        self.pq = queue.PriorityQueue()
        self.started = False
        self.numSeen = 0

    def __iter__(self):
        return self

    def __next__(self):
        if not self.started:
            self.started = True
            #print("1put", (self.valueFunction([self.members[0]]), [0]))
            self.pq.put((self.f(np.array([self.members[0]])), self.numSeen, np.array([0])))
            self.numSeen += 1
            return np.array([])

        if self.pq.empty():
            raise StopIteration()

        # get the next best subset
        nextSubset = self.pq.get()[2]

        # move this subset to look for the next "last member"
        if nextSubset[-1] < self.members.size-1:
            nextSubsetInc = copy.copy(nextSubset)
            nextSubsetInc[-1] += 1
            #print("2put", (self.valueFunction([self.members[i] for i in nextSubsetInc]), nextSubsetInc))
            self.pq.put((self.f(self.members[nextSubsetInc]), self.numSeen, nextSubsetInc))
            self.numSeen += 1


        # add the grown version of this subset
        if nextSubset[-1] < self.members.size-1:
            nextSubsetGrown = np.append(nextSubset, nextSubset[-1]+1)
            #print("3put", (self.f(self.members[nextSubsetGrown]), nextSubsetGrown))
            self.pq.put((self.f(self.members[nextSubsetGrown]), self.numSeen, nextSubsetGrown))
            self.numSeen += 1

        return self.members[nextSubset]

class ESKernelSubsets:
    def __init__(self, members, variances):
        self.M = members.size

        self.totalCount = 2**self.M
        self.count = 0

        # ensure all the variances are unique
        self.variances = variances + np.arange(self.M-1, -1, -1)/1e5

        def subsetValue(x):
            s = x.size
            if s == self.M or s == 0:
                return 1e12
            elif s > self.M/2:
                return -1
            elif s == 1:
                return 1e10 + min(variances[x])

            w = (self.M-1)/(scipy.special.binom(self.M,s)*s*(self.M-s))
            return min(variances[x])*w

        self.itr = OrderedSubsets(members, subsetValue)
        self.complement = False

    def eskernelSubsetWeight(self, s):
        if s == self.M or s == 0:
            return 1e12
        return (self.M-1)/(scipy.special.binom(self.M,s)*s*(self.M-s))

    def __next__(self):
        self.count += 1
        if self.count > self.totalCount:
            raise StopIteration()
        elif not self.complement:
            subsetElements = self.itr.__next__()
            self.lastSubset = subsetElements
            self.complement = True
            out = np.zeros(self.M)
            if subsetElements.size > 0:
                out[subsetElements] = 1
        else:
            out = np.ones(self.M)
            if self.lastSubset.size > 0:
                out[self.lastSubset] = 0
            self.complement = False

        return (out,self.eskernelSubsetWeight(sum(out)))

    def __iter__(self):
        return self
