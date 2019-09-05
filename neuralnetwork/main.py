from neuralnetwork import *

if __name__ == '__main__':
    X = numpy.array([
        [0, 0], 
        [0, 1], 
        [1, 0], 
        [1, 1], 
    ])
    T = numpy.array([
        [0], [1], [1], [0],
    ])
    N = X.shape[0]

    input_size = X.shape[1]
    hidden_size = 2
    output_size = 2
    epsilon = 0.1
    mu = 0.9
    epoch = 10000

    nn = Neural(input_size, hidden_size, output_size)
    nn.train(X, T, epsilon, mu, epoch)
    nn.error_graph()

    C, Y = nn.predict(X)

    for i in range(N):
        x, t, c = X[i, :], T[i, :], C[i]
        print("x: {}\n".format(x))
        print("t: {}\n".format(t))
        print("c: {}\n".format(c))
