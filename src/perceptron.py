import numpy as np


class Perceptron(object):
    """ Implementation of Rosenblatt's Perceptron """
    def __init__(self, seed:int=0, epochs:int=50, learning_rate:float=0.01):
        """ Initialization of Perceptron Parameters """
        self.seed = seed
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.w = None

    def fit(self, X:np.array, y:np.array):
        """ Return the trained perceptron.

        Learning the perceptron using the Delta rule

        Keyword arguments:\n
            X : object features\n
            y : the correct class label
        """
        X = self.__make_bias(X).astype(np.float32)

        rnd = np.random.RandomState(self.seed)
        self.w = rnd.random(size=X.shape[1]).astype(np.float32)  # initializing the initial weights
        
        for _ in range(self.epochs):
            for x, y_train in zip(X, y):
                error = y_train - self.__activate(x, self.w)
                for j in range(3):
                    self.w[j] = self.w[j] + self.learning_rate * error * x[j]  # Delta rule

        return self

    def predict(self, x):
        """ Return label predicted by the model
        Prediction function

        Keyword arguments:\n
            x : input features
        """
        x = self.__make_bias(x)
        return self.__activate(x, self.w)

    def __make_bias(self, X):
        """ Adding an bias """
        bias = np.ones((np.array(X).shape[0], 1))
        return np.concatenate((np.array(X), bias), axis=1)

    def __activate(self, x, w) -> int:
        """ Activation function """
        if len(x.shape) > 1:
            res = list()
            for x_ in x:
                if (x_ @ w.T) >= 0:
                    res.append(1)
                else: 
                    res.append(0)
            return np.array(res).astype(np.float32)

        if (x @ w.T) >= 0: return np.array(1).astype(np.float32)
        else: return np.array(0).astype(np.float32)
