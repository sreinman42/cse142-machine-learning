import random

def L(x, y, p):
    assert(len(x) == len(y))
    d = 0
    for i in range(len(x)):
        d = d + (abs(x[i] - y[i]))**p
        d = d**(1/p)
    return d

def main():
    x = [33.6, 30.6, 4.8, 6.8, 1.22, 2.11, 3.00]
    y = [36.7, 27.0, 4.7, 11.3, 1.0, 1.67, 3.83]
    print(x, y)
    for p in [1, 2, 10, 100]:
        print(L(x, y, p))

    v = [5, 5, 2, 2, 0.5, 0.1, 1]
    x_v = [a + b for a, b in zip(x, v)]
    y_v = [a + b for a, b in zip(y, v)]
    print(x_v, y_v)
    for p in [1, 2, 10, 100]:
        print(L(x_v, y_v, p))

    k = random.randint(2, 10)
    x_k = [a * k for a in x]
    y_k = [a * k for a in y]
    print(x_k, y_k)
    for p in [1, 2, 10, 100]:
        print(L(x_k, y_k, p))

main()
