import numpy as np

def f(x):
    return np.vectorize(lambda x : 1 if x >= 0 else -1)(x)

def feed_forward(w, x):
    return f(np.dot(w, x))

def train(w, x, t, eps):
    y = feed_forward(x, w)
    if(t != y):
        w += eps * t * x
    return w


def main():
    eps = 0.1
    max_epoch = 100

    X = np.array([[1,0,0], [1,0,1], [1,1,0], [1,1,1]], dtype=np.float32)
    t = np.array([-1,-1,-1,1], dtype=np.float32)

    w  = np.array([0,0,0], dtype=np.float32)

    for e in range(max_epoch):
        for i in range(t.size):
            w = train(w, X[i], t[i], eps)

    y = f(np.sum(w*X, 1))
    print("y: ", y)
    print("t: ", t)
    print("w: ", w)

if __name__ == "__main__":
    main()
