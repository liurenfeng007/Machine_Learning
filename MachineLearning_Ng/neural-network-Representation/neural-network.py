import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.optimize import minimize


# logic函数g
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

#代价函数J
def cost(theta, X, y, learningRate):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    first = np.multiply(-y, np.log(sigmoid(X * theta.T)))
    second = np.multiply((1 - y), np.log(1 - sigmoid(X * theta.T)))
    reg = (learningRate / (2 * len(X))) * np.sum(np.power(theta[:,1:theta.shape[1]], 2))
    J=np.sum(first - second) / len(X) + reg
    return J


#梯度下降，但是在theta0或者theta需要进行判断
def gradientDescent(theta, X, y,lambda1):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)

    parameters = int(theta.ravel().shape[1])
    grad = np.zeros(parameters)
    error = sigmoid(X * theta.T) - y
    for i in range(parameters):
        term = np.multiply(error, X[:, i])
        if (i == 0):
            grad[i] = np.sum(term) / len(X)
        else:
            grad[i] = (np.sum(term) / len(X)) + ((lambda1 / len(X)) * theta[:, i])
    return grad

#构建分类器,由于逻辑回归只能一次在2个类之间进行分类，我们需要多类分类
def one_vs_all(X, y, num_labels, learning_rate):
    rows = X.shape[0]
    params = X.shape[1]

    # k X (n + 1) array for the parameters of each of the k classifiers
    all_theta = np.zeros((num_labels, params + 1))

    # insert a column of ones at the beginning for the intercept term
    X = np.insert(X, 0, values=np.ones(rows), axis=1)

    # labels are 1-indexed instead of 0-indexed
    for i in range(1, num_labels + 1):
        theta = np.zeros(params + 1)
        y_i = np.array([1 if label == i else 0 for label in y])
        y_i = np.reshape(y_i, (rows, 1))
        print("正在生成第{}个分类器...".format(i))
        # minimize the objective function
        fmin = minimize(fun=cost, x0=theta, args=(X, y_i, learning_rate), method='TNC', jac=gradientDescent)
        # print(fmin)
        # print(fmin.x)
        all_theta[i - 1, :] = fmin.x
    return all_theta # 10*401


def predict_all(X, all_theta):
    rows = X.shape[0]
    # same as before, insert ones to match the shape
    X = np.insert(X, 0, values=np.ones(rows), axis=1)
    # convert to matrices
    X = np.matrix(X)
    all_theta = np.matrix(all_theta)
    # compute the class probability for each class on each training instance
    h = sigmoid(X * all_theta.T)#5000*401  *  401*10 = 5000*10
    print(h)
    # create array of the index with the maximum probability
    h_argmax = np.argmax(h, axis=1)
    print(h_argmax)
    # because our array was zero-indexed we need to add one for the true label prediction
    h_argmax = h_argmax + 1

    return h_argmax

#抽取一张图片，维度400=20*20
def plot_an_image(image):
    fig, ax = plt.subplots(figsize=(1, 1))
    ax.matshow(image.reshape((20, 20)), cmap=plt.cm.binary)
    plt.xticks(np.array([]))  # just get rid of ticks
    plt.yticks(np.array([]))

#抽取100张图片
def plot_100_image(X):
    size = int(np.sqrt(X.shape[1]))

    # sample 100 image, reshape, reorg it
    sample_idx = np.random.choice(np.arange(X.shape[0]), 100)  # 100*400
    sample_images = X[sample_idx, :]

    fig, ax_array = plt.subplots(nrows=10, ncols=10, sharey=True, sharex=True, figsize=(8, 8))

    for r in range(10):
        for c in range(10):
            ax_array[r, c].matshow(sample_images[10 * r + c].reshape((size, size)),
                                   cmap=plt.cm.binary)
            plt.xticks(np.array([]))
            plt.yticks(np.array([]))
            #绘图函数，画100张图片




data = loadmat('ex3data1')
X=data.get('X')
y=data.get('y')
#print(y)

#随意展示一张图片
pick_one = np.random.randint(0, 5000)
plot_an_image(X[pick_one, :])
plt.show()
print('this should be {}'.format(y[pick_one]))
#展示一百张图片
plot_100_image(X)
plt.show()


print(data['X'].shape, data['y'].shape)  # X:5000*400,Y:5000*1
all_theta = one_vs_all(data['X'], data['y'], 10, 1)
y_pred = predict_all(data['X'], all_theta)
correct = [1 if a == b else 0 for (a, b) in zip(y_pred, data['y'])]
accuracy = (sum(map(int, correct)) / float(len(correct)))
print ('accuracy = {0}%'.format(accuracy * 100))
