import numpy as np

train_X = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
train_y = np.array([[0, 1], [0, 0], [1, 0], [1, 1]])
weights = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])


def step(value):
    return 1.0 if value >= 0.0 else 0.0


def forward(X, weights):
    y = [weights[0].dot(np.r_[X, np.array([1])]), weights[1].dot(np.r_[X, np.array([1])])]
    return np.vectorize(step)(y)


def train(weights):
    eta = 0.01
    for X, y in zip(train_X, train_y):
        o = forward(X, weights)
        for k in range(len(train_y[0])):
            weights[k] += (y[k] - o[k])*eta*np.r_[X, np.array([1])]

        #  print("weights:\n{}".format(weights))
    return weights


if __name__ == '__main__':
    for i in range(len(train_X)):
        y = forward(train_X[i], weights)
        print(y)
    print("====================")

    for i in range(100):
        weights = train(weights)

    for i in range(len(train_X)):
        y = forward(train_X[i], weights)
        print(y)

