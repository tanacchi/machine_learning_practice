import numpy
import math
import random
import matplotlib
from matplotlib import pyplot

class Neural(object):

    def __init__(self, input_num, hidden_num, output_num):
        self.hidden_weight_matrix = numpy.random.random_sample((hidden_num, input_num+1))
        self.output_weight_matrix = numpy.random.random_sample((output_num, hidden_num+1))
        self.hidden_momentum = numpy.zeros((hidden_num, input_num+1))
        self.output_momentum = numpy.zeros((output_num, input_num+1))


    def train(self, X, T, epsilon, mu, epoch):
        self.error = numpy.zeros(epoch)
        N = X.shape[0]
        for epo in range(epoch):
            for i in range(N):
                x, t = X[i, :], T[i, :]
                self.__update_weight(x, t, epsilon, mu)
            self.error[epo] = self.__calc_error(X, T)

    def predict(self, X):
        N = X.shape[0]
        C = numpy.zeros(N).astype('int')
        Y = numpy.zeros((N, X.shape[1]))
        for i in range(N):
            x = X[i, :]
            z, y = self.__forward(x)
            Y[i] = y
            C[i] = y.argmax()
        return (C, Y)

    
    def error_graph(self):
        matplotlib.use('TkAgg')
        pyplot.ylim(0.0, 2.0)
        pyplot.plot(numpy.arange(0, self.error.shape[0]), self.error)
        pyplot.show()


    def __sigmoid(self, arr):
        return numpy.vectorize(lambda x: 1.0 / (1.0 + math.exp(-x)))(arr)


    def __forward(self, x):
        z = self.__sigmoid(self.hidden_weight_matrix.dot(numpy.r_[numpy.array([1]), x]))
        y = self.__sigmoid(self.output_weight_matrix.dot(numpy.r_[numpy.array([1]), z]))
        return (z, y)

    def __update_weight(self, x, t, epsilon, mu):
        z, y = self.__forward(x)
        output_delta = (y - t) * y * (1.0 - y)
        _output_weight_matrix = self.output_weight_matrix
        self.output_weight_matrix -= epsilon * output_delta.reshape((-1, 1)) * numpy.r_[numpy.array([1]), z] - mu * self.output_momentum
        self.output_momentum = self.output_weight_matrix - _output_weight_matrix

        hidden_delta = (self.output_weight_matrix[:, 1:].T.dot(output_delta)) * z * (1.0 - z)
        _hidden_weight_matrix = self.hidden_weight_matrix
        self.hidden_weight_matrix -= epsilon * hidden_delta.reshape((-1, 1)) * numpy.r_[numpy.array([1]), x]
        self.hidden_momentum = self.hidden_weight_matrix - _hidden_weight_matrix


    def __calc_error(self, X, T):
        N = X.shape[0]
        err = 0.0
        for i in range(N):
            x, t = X[i, :], T[i, :]
            z, y = self.__forward(x)
            err += (y - t).dot((y - t).reshape((-1, 1))) / 2.0
        return err
    

if __name__ == '__main__':
    a = numpy.random.random_sample((5,2))
    print("random_sample:\n{}\n".format(a))
    print("zeros:\n{}\n".format(numpy.zeros((5, 2))))
    print("vectorize:\n{}\n".format(numpy.vectorize(lambda x: x*2)(a)))
    print("r_:\n{}\n".format(numpy.r_[numpy.array([1]), numpy.array([2, 3])]))
    print("a[1, :]:\n{}\n".format(a[1, :]))
    print("a.shape[0]:\n{}\n".format(a.shape[0]))
    print("a.shape[1]:\n{}\n".format(a.shape[1]))
