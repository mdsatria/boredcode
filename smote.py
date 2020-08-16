import numpy as np

class smote():
    def __init__(self, data, N, k=3):
        self.data = data
        self.N = N
        self.k = k
        self.n, self.m = data.shape
        
    def distance(self, a, b):
        return np.sqrt(np.sum((a-b)**2, axis=0))
    
    def knn(self, instance):
        temp = np.zeros(self.n)
        for i in range(self.n):
            temp[i] = self.distance(instance, self.data[i, :])
        
        return np.argsort(temp)[:self.k]
    
    def populateArray(self):
        temp = np.zeros(shape=(self.n, self.k))
        for i in range(self.n):
            temp[i, :] = np.array(self.knn(self.data[i, :]))
            
        return temp.astype(int)
    
    def newData(self):
        temp = np.zeros(shape=(self.n*self.N, self.m))
        nnArray = self.populateArray()
        l = 0
        for i in range(self.n):
            for j in range(self.N):
                temp[l, :] = self.data[i, :] + ((self.data[nnArray[i, self.N], :]) * np.random.uniform(0, 1))
                l += 1
        
        return temp
    
# x = np.random.randint(0, 100, size=(10, 5))
# a = smote(x,2)
# a.newData()