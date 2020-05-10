
import numpy as np

class SVM(object):

    def __init__(self, X, y, var=1.0, learnRate=1, xi=10):
        """
        Initializes an empty model
        """
        # cache data and kernel
        self.X = X
        self.y = y

        # initialize a value of alpha that satisfies sum(a.*y) = 0
        self.alpha = np.random.uniform(size=(X.shape[0]))
        self.alpha[self.y ==  1] /= np.sum(self.alpha[self.y ==  1])
        self.alpha[self.y == -1] /= np.sum(self.alpha[self.y == -1])

        self.var = var
        self.learnRate = learnRate
        self.xi = xi
        self.w0 = 0
        self.K = self.__rbf(np.matmul(X, X.T))

    def train(self, epochs):
        """
        """
        for i in range(epochs):

            if(i % 1000 == 0):
                print(self.__loss())

            self.__sgd_one()

        self.w0 = (1 - self.xi)

    def predict(self, X0):
        """
        """
        X0Dots = np.matmul(X0, self.X.T)
        X0Kern = self.__rbf(X0Dots)

        ay = np.multiply(self.alpha, self.y)
        ayK = np.multiply(X0Kern, ay)

        #print(np.sum(ayK, axis=1))
        y0 = np.sign( np.sum(ayK, axis=1) + self.w0 )
        return y0

    def __sgd_one(self):
        """
        Take a single step in the direction of the
        gradient of the dual loss function.

        L = sum(ai) - (1/2) * sum_i(sum_j(ai*yi*aj*yj*K(i,j)))
        dL/dai = 1 - yi *sum_j(aj*yj*K(i,j))
        """

        # calc dL/dAlpha
        ay = np.multiply(self.alpha, self.y)
        ayK = np.multiply(self.K, ay)
        print(ayK)

        dL_dAlpha = np.ones(self.y.shape) - np.multiply(self.y, np.sum(ayK, axis=1))
        dL_dAlpha[dL_dAlpha < 0] = 0

        # update alpha and constrain
        self.alpha -= (self.learnRate * dL_dAlpha)
        
    def __loss(self):
        """
        """
        ay = np.multiply(self.alpha, self.y)
        ayMat = np.outer(ay, ay)
        ayKMat = np.multiply(ayMat, self.K)

        L = np.sum(self.alpha) - 0.5 * np.sum(ayKMat)
        return L

    def __rbf(self, mat):
        """
        applies the rbf kernel to to a given input matrix
        """
        return np.exp(-1 * mat / self.var)