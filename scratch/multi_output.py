train_X = [[0.0, 0.0, 1.0], [0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0]]
train_y = [[0, 1], [1, 1], [0, 0], [1, 0]]
weights = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]

def get(X):
    return  [X[0]*weights[0][0] + X[1]*weights[0][1] + X[2]*weights[0][2],
             X[0]*weights[1][0] + X[1]*weights[1][1] + X[2]*weights[1][2]]


def step(value):
    return 1.0 if value >= 0.0 else 0.0


def forward(X):
    func = lambda x : 1.0 if x >= 0.0 else 0.0
    tmp = get(X)
    return [func(tmp[0]), func(tmp[1])]



def train():
    episode = 100
    eta = 0.01

    for epi in range(episode):
        for i in range(len(train_X)):
            X, y = train_X[i], train_y[i]
            o = forward(X)
            for j in range(len(train_X[0])):
                for k in range(len(train_y[0])):
                    weights[k][j] += (y[k] - o[k])*eta*X[j]
        print("weights: {}".format(weights))


if __name__ == '__main__':
    for i in range(len(train_X)):
        y = forward(train_X[i])
        print(y)

    train()

    for i in range(len(train_X)):
        y = forward(train_X[i])
        print(y)

