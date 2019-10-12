inputs  = [[0.0, 0.0, 1.0], [0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0]]
answers = [1, 0, 1, 1]
weights = [0.0, 0.0, 0.0]

def get(X):
    return X[0]*weights[0] + X[1]*weights[1] + X[2]*weights[2]

def step(value):
    return 1.0 if value > 0.0 else 0.0

def forward(X):
    return step(get(X))

def train():
    episode = 100
    eta = 0.1

    for epi in range(episode):
        for i in range(len(inputs)):
            X, y = inputs[i], answers[i]
            o = forward(X)
            for j in range(len(inputs[0])):
                weights[j] += (y - o)*eta*X[j]
    print("weights: {}".format(weights))



if __name__ == '__main__':
    for i in range(len(inputs)):
        y = forward(inputs[i])
        print(y)

    train()

    for i in range(len(inputs)):
        y = forward(inputs[i])
        print(y)
