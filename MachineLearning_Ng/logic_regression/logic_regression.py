import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as opt
import time

#时间装饰
def Time(func):
    def newFunc(*args, **args2):
        t0 = time.time()
        back = func(*args, **args2)
        return back, time.time() - t0
    return newFunc


def loadData(filename):
    if filename=='ex2data1.txt':
        data = pd.read_csv(filename, header=None, names=['Exam 1', 'Exam 2', 'Admitted'])
    else:
        data = pd.read_csv(filename, header=None, names=['Test 1', 'Test 2', 'Accepted'])
        # 标准化
        #data = (data - data.mean()) / data.std()
        # 归一化 data = (data - data.min()) / (data.max()-data.min())
    data.insert(0, 'Ones', 1)
    # print(data)
    X = data.iloc[:, :-1]  # X是所有行，去掉最后一列，即特征
    y = data.iloc[:, -1]  # y是所有行，最后一列，即特征
    # 代价函数是应该是numpy矩阵，所以我们需要转换X和Y，然后才能使用它们。 我们还需要初始化theta。
    X = np.array(X.values)#m*n
    y = np.array(y.values).reshape(X.shape[0], 1)#m*1
    return X, y,data


def sigmoid(z):  # 逻辑函数
    g=1.0/(1.0+np.exp(-z))
    return g


def hypothesis(theta, X):  # 模型函数
    h=sigmoid(X* theta)
    return h


def predict(theta, X):
    probability = sigmoid(X * theta)
    return [1 if x >= 0.5 else 0 for x in probability]


def costfunction(X, y, theta):  #d代价函数
    m=len(X)
    first = np.multiply(-y, np.log(sigmoid(X * theta)))
    second = np.multiply((1 - y), np.log(1 - sigmoid(X * theta)))
    J = np.sum(first - second) / m
    return float(J)

def batch_gradientDescent(X, y, alpha, iteration):
    theta = np.matrix(np.zeros([X.shape[1],1]))#n*1
    cost = np.zeros(iteration)
    thetas = {}
    for j in range(X.shape[1]):
        thetas[j] = [theta[j,0]]
    for i in range(iteration) :
        loss=sigmoid(X * theta) - y  # m*1
        gradient = (np.transpose(X) * loss)/len(X)  # n*1
        theta = theta-alpha*gradient
        for j in range(X.shape[1]):
            thetas[j].append(theta[j, 0])
        cost[i] = costfunction(X, y, theta)
    return theta,cost,thetas # n*1


alpha = 0.005
iteration = 1000
X, y,data=loadData('ex2data1.txt')
theta,cost,thetas=batch_gradientDescent(X, y, alpha, iteration)
print(theta)


# 绘制分布图
# 创建两个分数的散点图，并使用颜色编码来可视化，即正类、负类
positive = data[data['Admitted'].isin([1])]
negative = data[data['Admitted'].isin([0])]

fig, ax = plt.subplots(figsize=(12,8))
ax.scatter(positive['Exam 1'], positive['Exam 2'], s=50, c='b', marker='o', label='Admitted')
ax.scatter(negative['Exam 1'], negative['Exam 2'], s=50, c='r', marker='x', label='Not Admitted')
ax.legend()
ax.set_xlabel('Exam 1 Score')
ax.set_ylabel('Exam 2 Score')
plt.show()

# 逻辑回归模型
nums = np.arange(-20, 20)
fig, ax = plt.subplots(figsize=(12,8))
ax.plot(nums, sigmoid(nums), 'r')
plt.show()

#计算准确性
#theta_min = opt.fmin_tnc(func=costfunction, x0=theta, fprime=batch_gradientDescent, args=(X, y))
predictions = predict(theta, X)
correct = [1 if ((a == 1 and b == 1) or (a == 0 and b == 0)) else 0 for (a, b) in zip(predictions, y)]
accuracy = (sum(map(int, correct)) % len(correct))
print ('accuracy = {0}%'.format(accuracy))


from sklearn import linear_model#调用sklearn的线性回归包
model = linear_model.LogisticRegression(penalty='l2', C=1.0)
model.fit(X, y.ravel())
print ('accuracy = {0}%'.format(int(model.score(X, y)*100)))


X, y,data=loadData('ex2data2.txt')
theta,cost,thetas=batch_gradientDescent(X, y, alpha, iteration)

positive = data[data['Accepted'].isin([1])]
negative = data[data['Accepted'].isin([0])]

fig, ax = plt.subplots(figsize=(12,8))
ax.scatter(positive['Test 1'], positive['Test 2'], s=50, c='b', marker='o', label='Accepted')
ax.scatter(negative['Test 1'], negative['Test 2'], s=50, c='r', marker='x', label='Rejected')
ax.legend()
ax.set_xlabel('Test 1 Score')
ax.set_ylabel('Test 2 Score')
plt.show()
