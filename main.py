import numpy as np

from perceptron import Perceptron


def main():
    X = np.array([[0, 0, 1],
                  [1, 1, 1],
                  [1, 0, 1],
                  [0, 1, 1]])


    y = [0, 1, 1, 0]
    per = Perceptron()
    per.fit(X, y)
    predict1 = [[1, 0, 0]]
    print(per.predict(predict1))

if __name__ == '__main__':
    main()
