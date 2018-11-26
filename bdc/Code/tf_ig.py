import sys
sys.path.append('..')
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import math
import numpy as np

def tf_ig(corpus,test,package):
	
	labelset = package ["labelset"]
	weights = package ["weights"]
	doclist = package ["doclist"]
	dictlist = {}
	doclen = {}
	n = len(corpus)
	worddict = {}

	# labell : 目录名
	# label: 目录列表
	# dictlist: 目录名——词表 的三层词典嵌套， 以label为KEY索引到文本列表，每个文本是词语列表，词语列表包含该词在这篇文本中的频次
	# doclist: 目录名——词表——文档 的三层词典嵌套， 以label为KEY索引到词表，每个词对应包含的文档名集合
	# worddict : 词——文档 词典， 每个词对应 包含该词语的文档数目（不看label）

	for i in corpus:
		labell = i["label"]
		docl = i["document"]
		if not labell in doclen:
			doclen[labell] = 0
		doclen[labell] += i["length"]
		for j in i["split_sentence"]:
			# 计算 dictlist : label —— doc —— word —— frequency
			if not labell in dictlist:
				dictlist[labell] = {}
			if not j in dictlist[labell]:
				dictlist[labell][j] = 1
			else:
				dictlist[labell][j] += 1

			if test ==0 :
				# 计算 doclist : label —— word ——　doc set
				if not labell in doclist:
					doclist[labell] = {}
				if not j in doclist[labell]:
					doclist[labell][j] = set()
				doclist[labell][j].add(docl)

	if test ==0:
		# 获取每个词出现的文档数目（不看labell）
		# a = len(docllist[labell][word])
		# b = word出现的文档数目 - a
		# c = 用for求sum(doclist[label][!word])
		# d = sum( !word 出现的文档数目 ) - c
		# ig = -(a+c)/n*log((a+c)/n)+a/n*log(a/(a+b))+c/n*log(c/(c+d))

		for labell in labelset:
			for word in doclist[labell]:
				if not word in worddict:
					worddict[word] = 0
				worddict[word] += len(doclist[labell][word])

		# 计算 ig
		for labell in labelset:
			weights[labell] = {}
			for word in doclist[labell]:
				a_b = worddict[word]
				a = len(doclist[labell][word])
				b = a_b - a
				c_d = sum([(worddict[x]) for x in worddict.keys() if x!=word])
				c = sum([len(doclist[labell][x]) for x in (doclist[labell]) if x!=word])
				d = c_d - c
				weights[labell][word] = -1.0*( (a+c)*1.0/n )*math.log( (a+c)*1.0/n,2 ) + (a*1.0/n)*math.log( a*1.0/(a+b),2 ) + (c*1.0/n)*math.log( c*1.0/(c+d),2 )

	# 计算 tf-ig 值
	tf_ig = {}
	for labell in labelset:
		tf_ig[labell] = {}
		for word in dictlist[labell]:
			tf_ig[labell][word] = dictlist[labell][word]*1.0 /  (doclen[labell]*1.0)
			if word in weights[labell]:
				tf_ig[labell][word] *= weights[labell][word]
			else:
				tf_ig[labell][word] *= max([ weights[x][word]  for x in weights if word in weights[x]])
	package ["labelset"] = labelset
	package ["weights"] = weights
	package ["doclist"] = doclist

	return tf_ig
