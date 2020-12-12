import numpy as np

from perceptron import Perceptron


def main():
    X = np.array([[0, 0, 1, 3],
                  [1, 1, 1, 8],
                  [1, 0, 1, 4],
                  [0, 1, 1, 2]])


    y = [0, 1, 1, 0]
    model = Perceptron()
    model.fit(X, y)
    predict = np.array([[0, 0, 1, 0]]).astype(np.float32)
    out = per.predict(predict)
    print(out)

if __name__ == '__main__':
    main()
