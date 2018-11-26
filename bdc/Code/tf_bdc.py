import sys
sys.path.append('..')
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import math
import numpy as np

def tf_bdc(corpus,test,package):

	labelset = package ["labelset"]
	weights = package ["weights"]
	voca = package ["voca"]

	label_text = {}
	doclen = {}
	dictlist = {}

	# labell : 类名
	# label_text: 类名——文本 的词典， 文本是语料库中所有有同样类名的句子的总和。
	for i in corpus:
		labell = i["label"]
		sentence = i["split_sentence"]
		if not labell in label_text:
			label_text[labell] = []
		label_text[labell] = label_text[labell] + sentence

	# 计算 tf的预备
	# doclen: 每个目录下的总词数
	# dictlist: 每个目录下每个词的频率，以列表形式保存，列表内以 词名——频次 词典保存
	for a in label_text:
		listt = label_text[a]
		doclen[a]= len(listt)
		if not a in dictlist:
			dictlist[a] = {}
		for i in listt:
			if not i in dictlist[a]:
				dictlist[a][i] = 0
			dictlist[a][i] += 1

	# 计算 doclist
	if test ==0:
		for word in voca:
			hlist = np.zeros((len(labelset)))
			#print len(hlist)
			for cate in range(len(labelset)):
				if word in dictlist[labelset[cate]]:
					hlist[cate] = dictlist[labelset[cate]][word]*1.0/doclen[labelset[cate]]
			hlist = hlist/np.sum(hlist)
			for i in range(len(hlist)):
				if abs(hlist[i]-0.0)<1e-5:
					hlist[i]=1
			weights[word] = ( np.sum( hlist*np.log2(hlist)) ) / (math.log((len(labelset)+1),2))

	# 计算 tf-dc 值
	for i in dictlist:
		for j in dictlist[i]:
			dictlist[i][j] = dictlist[i][j] * 1.0 / doclen[i]
			dictlist[i][j] = dictlist[i][j] * weights[j]
	package ["labelset"] = labelset
	package ["weights"] = weights
	package ["voca"] = voca
	return dictlist
