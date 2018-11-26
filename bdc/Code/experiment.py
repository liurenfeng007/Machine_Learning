import sys
sys.path.append('..')
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import numpy as np
import matplotlib.pyplot as plt 
from sklearn import preprocessing
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier as KNN
from one_hot import one_hot
from preprocess import preprocess
from tf_idf import tf_idf
from tf_dc import tf_dc
from tf_bdc import tf_bdc
from tf_ig import tf_ig
from tf_eccd import tf_eccd
from tf_chi import tf_chi
from tf_rf import tf_rf
from iqf_qf_icf import iqf_qf_icf

def init(package):
    package["voca"] = []               #单词表
    package["labelset"] = []           #标签表
    package["vocafreq"] = {}           #词语——词频的字典
    package["weights"] = {}            #词语——权重的字典
    package["doclist"] = {}            #目录名——词表——文档 的三层词典嵌套， 以label为KEY索引到词表，每个词对应包含的文档名集合
    package["docname"] = set()         #创造一个集合，即无重复的标签集


    #参数介绍：
    # input: 预处理过的语料库
    # algo: 使用的特征权重计算方法名
    # model: 使用的模型名
    # test = 0 : 记录文件中出现的词汇并构造词汇表(训练集)
    # test = 1 : 不构造词汇表，用已经构造好的(测试集)
def getXY(input, algo, model, test=0):
    global package
    corpus = preprocess(input, package, test)
    labelset = package["labelset"]
    voca = package["voca"]
    weights={}
    level = 2
    mod = 0
    if algo == "tf_idf":
        weights = tf_idf(corpus,test,package)
        mod=1
    elif algo == "tf_dc":
        weights = tf_dc(corpus,test,package)
    elif algo == "tf_bdc":
        weights = tf_bdc(corpus,test,package)
    elif algo == "iqf_qf_icf":
        weights = iqf_qf_icf(corpus,test,package)
    elif algo == "tf_eccd":
        weights = tf_eccd(corpus,test,package)
    elif algo == "tf_ig":
        weights = tf_ig(corpus,test,package)
    elif algo == "tf_rf":
        weights = tf_rf(corpus,test,package)
        level = 3
    elif algo == "tf_chi":
        weights = tf_chi(corpus,test,package)
        level = 3
    #print weights
    X = []
    Y = []
    count = 0
    vocalen = len(voca)
    for doc in corpus:
        if count%100 ==0:
            print(str(count) + "/" + str(len(corpus)))
        count+=1

        # process label
        labelset.append(doc["label"])
        Y.append(int(np.argmax(one_hot(labelset)[-1])))
        labelset = labelset[:-1]
        #print(Y)

        # process word
        temvocalist = voca + doc["split_sentence"]
        tem_one_hot = one_hot(temvocalist)[vocalen:]
        for word in range(len(tem_one_hot)):
            temlabel = doc["label"]
            temword = doc["split_sentence"][word]
            temdoc = doc["document"]
            if level == 2:
                if mod ==0:
                    tem_one_hot[word] *= weights[temlabel][temword]
                else:
                    tem_one_hot[word] *= weights[temdoc][temword]
            else:
                tem_one_hot[word] *= weights[temlabel][temdoc][temword]

        tem_one_hot = np.max(tem_one_hot,axis=0)
        #if (model.lower()=="knn"):
        tem_one_hot = preprocessing.normalize(np.array(tem_one_hot).reshape(1,-1), norm='l2')
        X.append(tem_one_hot)
        #print(X)
    #print(Y)

    return np.squeeze(X),Y


    # algo: 可选 tf_idf, tf_dc, tf_bdc, tf_ig, tf_chi, tf_eccd, tf_rf, iqf_qf_icf
    # model: 可选 svm, knn
    # knn_neighbour:
    # 为0：测试模式，选用 [1,5,10,15,20,25,30,35] 作为邻居数分别进行训练，文件输出正确率（可plot或导入evaluate程序）
    # 为n：使用邻居数为n进行训练，文件输出所有预测标签和正确标签，中间以\t分离
def main( trainf, testf , algo="tf_idf",model="knn",knn_neighbour=0):
    global package
    package = {}
    init(package)

    print("Training "+model+" with "+algo)
    print("Processing Training Set... ")
    train_x, train_y = getXY(trainf,algo,model, test=0)
    print("Finish! ")
    print("Processing Test Set... ")
    test_x, test_y = getXY(testf,algo,model,test=1)
    print("Finish! ")

    if model =="svm":
        A=[]
        clf = svm.LinearSVC(penalty='l1',dual=False)
        clf.fit(train_x,train_y)
        res = clf.predict(np.array(test_x))

        resultfile = open(model + "_" + algo+".txt","w")
        for i in range(len(res)):
            resultfile.write( str(res[i] ) +" "+ str( test_y[i] )+"\n" )
        resultfile.close()

    else:
        if knn_neighbour!=0:
            clf = KNN(n_neighbors = knn_neighbour , weights='uniform')
            clf.fit(train_x,train_y)
            result = clf.predict(np.array(test_x))
            resultfile = open(model + "_" + algo+".txt","w")
            for i in range(len(result)):
                resultfile.write( str( result[i] ) +" "+ str( test_y[i] )+"\n" )
            resultfile.close()

        else:
            X = [1]+range(5,76,5)
            Y = []
            resultfile = open(model + "_" + algo+"test.txt","w")
            for i in X:
                clf = KNN(n_neighbors = i , weights='uniform')
                clf.fit(train_x,train_y)
                result = clf.predict(np.array(test_x))
                accuracy = sum(result==test_y)*1.0/len(test_y)
                Y.append(accuracy)

                str_nei = str(i)
                print(str_nei)
                print(accuracy)
                resultfile.write( str_nei+"\t"+str(accuracy)+"\n" )
            resultfile.close()
            #plt.plot(X,Y)
            #plt.show()



