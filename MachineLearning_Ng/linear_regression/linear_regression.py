from sklearn import linear_model
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.ticker as mtick
import time
from mpl_toolkits.mplot3d import Axes3D
from sklearn import linear_model


# 一元线性回归
# 耗时装饰器
def Time(func):
    def newFunc(*args, **args2):
        time_start=time.time()
        return func(*args, **args2), time.time()-time_start
    return newFunc


def loadData(filename):
    if filename=='ex1data1.txt':
        data = pd.read_csv(filename, header=None, names=['Population', 'Profit'])
    else:
        data = pd.read_csv(filename, header=None, names=['Size', 'Bedrooms', 'Price'])
        # 标准化
        data = (data - data.mean()) / data.std()
        # 归一化 data = (data - data.min()) / (data.max()-data.min())
    data.insert(0, 'Ones', 1)
    # print(data)
    X = data.iloc[:, :-1]  # X是所有行，去掉最后一列，即特征
    y = data.iloc[:, -1]  # y是所有行，最后一列，即特征
    # 代价函数是应该是numpy矩阵，所以我们需要转换X和Y，然后才能使用它们。 我们还需要初始化theta。
    X = np.matrix(X.values)#m*n
    y = np.matrix(y.values).reshape(X.shape[0], 1)#m*1
    return X, y,data


def hypothesis(theta, X):
    h=X* theta
    return h

def costfunction(X, y, theta):  #d代价函数
    m=len(X)
    J = (np.transpose(X * theta - y)) * (X * theta - y) / (2 * m)
    return float(J)


# 梯度下降BGD，可以尝试添加SGD、MBGD
def batch_gradientDescent(X, y, alpha, iteration):
    theta = np.matrix(np.zeros([X.shape[1],1]))#n*1
    cost = np.zeros(iteration)
    thetas = {}
    for j in range(X.shape[1]):
        thetas[j] = [theta[j,0]]
    for i in range(iteration) :
        loss=(X * theta) - y  # m*1
        gradient = (np.transpose(X) * loss)/len(X)  # n*1
        theta = theta-alpha*gradient
        for j in range(X.shape[1]):
            thetas[j].append(theta[j, 0])
        cost[i] = costfunction(X, y, theta)
    return theta,cost,thetas # n*1

# 正规方程
def normalEqation(X, y):
    theta = np.linalg.inv(X.T.dot(X)).dot((X.T).dot(y))#X.T@X等价于X.T.dot(X)
    return theta


#一元初始化
alpha = 0.01
iteration = 1000
X, y,data=loadData('ex1data1.txt')
theta,cost,thetas=batch_gradientDescent(X, y, alpha, iteration)
print('梯度下降theta:')
print(theta)
theta2=normalEqation(X, y)#和批量梯度下降的theta的值有点差距
print('正规方程theta:')
print(theta2)

#绘制分布图
data.plot(kind='scatter', x='Population', y='Profit', figsize=(12,8))
plt.savefig("Distribution.png")
plt.show()

#绘制拟合曲线
h=hypothesis(theta, X)
fig, ax = plt.subplots(figsize=(12,8))
ax.plot(data.Population, h, 'r', label='Prediction')
ax.scatter(data.Population, data.Profit, label='Traning Data')
ax.legend(loc=0)
ax.set_xlabel('Population')
ax.set_ylabel('Profit')
ax.set_title('Predicted Profit vs. Population Size')
plt.savefig("Fitting_Curve.png")
plt.show()

#下面我们进化为调包侠,使用scikit-learn做
model=linear_model.LinearRegression()
model.fit(X, y)
x = np.array(X[:, 1].A1)
h = model.predict(X)
fig, ax = plt.subplots(figsize=(12,8))
ax.plot(x, h, 'r', label='Prediction')
ax.scatter(data.Population, data.Profit, label='Traning Data')
ax.legend(loc=2)
ax.set_xlabel('Population')
ax.set_ylabel('Profit')
ax.set_title('Predicted Profit vs. Population Size')
plt.show()

#绘制误差曲线
fig, ax = plt.subplots(figsize=(12,8))
ax.plot(np.arange(iteration), cost, 'r')
ax.set_xlabel('Iterations')
ax.set_ylabel('Cost')
ax.set_title('Error vs. Training Epoch')
plt.savefig("Error.png")
plt.show()

# 绘制能量下降曲面
size = 100
theta0Vals = np.linspace(-10, 10, size)
theta1Vals = np.linspace(-2, 4, size)
JVals = np.zeros((size, size))
for i in range(size):
    for j in range(size):
        col = np.matrix([[theta0Vals[i]], [theta1Vals[j]]])
        JVals[i, j] = costfunction(X, y,col)

theta0Vals, theta1Vals = np.meshgrid(theta0Vals, theta1Vals)
JVals = JVals.T
contourSurf = plt.figure()
ax = contourSurf.gca(projection='3d')

ax.plot_surface(theta0Vals, theta1Vals, JVals, rstride=2, cstride=2, alpha=0.3,
                cmap=cm.rainbow, linewidth=0, antialiased=False)
ax.plot(thetas[0], thetas[1], 'rx')
ax.set_xlabel(r'$\theta_0$')
ax.set_ylabel(r'$\theta_1$')
ax.set_zlabel(r'$J(\theta)$')
plt.savefig("Energy_Gradient.png")
plt.show()

# 绘制能量轮廓
contourFig = plt.figure()
ax = contourFig.add_subplot(111)
ax.set_xlabel(r'$\theta_0$')
ax.set_ylabel(r'$\theta_1$')
CS = ax.contour(theta0Vals, theta1Vals, JVals, np.logspace(-2, 3, 20))
plt.clabel(CS, inline=1, fontsize=10)
   # 绘制最优解
ax.plot(theta[0, 0], theta[1, 0], 'rx', markersize=10, linewidth=2)
   # 绘制梯度下降过程
ax.plot(thetas[0], thetas[1], 'rx', markersize=3, linewidth=1)
ax.plot(thetas[0], thetas[1], 'r-')
plt.savefig("Energy_Contour.png")
plt.show()

#多元初始化
alpha = 0.01
iteration = 1000
X, y,data=loadData('ex1data2.txt')
theta,cost,thetas=batch_gradientDescent(X, y, alpha, iteration)

#多元误差曲线
fig, ax = plt.subplots(figsize=(12,8))
ax.plot(np.arange(iteration), cost, 'r')
ax.set_xlabel('Iterations')
ax.set_ylabel('Cost')
ax.set_title('Error vs. Training Epoch')
plt.savefig("Multiple_Error.png")
plt.show()





