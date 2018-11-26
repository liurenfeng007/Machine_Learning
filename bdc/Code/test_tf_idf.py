import sys
import os
import math
sys.path.append('..')
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

#tf·idf=（用词i出现的次数 / 词总数）*（log（文件总数/1+词语出现文件数))
def tf_idf(corpus, test , package):

	dictlist = {}
	doclen = {}
	docname = package ["docname"]
	weights = package ["weights"]

	for i in corpus:
		label = i["label"]
		document = i["document"]
		doclen[document] = i["length"]
		for j in i["split_sentence"]:
			# 计算 dictlist : document —— word —— frequency
			if not label in dictlist:
				dictlist[label] = {}
			if not j in dictlist[label]:
				dictlist[label][j] = 0
			dictlist[label][j] += 1
			if test==0:
				# 计算 doclist :  word ——　doc set
				if not j in weights:
					weights[j] = set()
				weights[j].add(document)
				docname.add(document)
	if test ==0:
		# 计算 idf 值
		for word in weights:
			weights[word] = math.log( ( len(docname)*1.0)/(len(weights[word])*1.0+1),2)

	# 计算 tf-idf 值
	tf_idf = {}
	for doc in dictlist:
		tf_idf[doc] = {}
		for word in dictlist[doc]:
			# tf:
			tf_idf[doc][word] = dictlist[doc][word]*1.0 / (doclen[doc]*1.0)
			# tf*idf
			tf_idf[doc][word] *= weights[word]
	package ["docname"] = docname
	package ["weights"] = weights
	#print tf_idf
	return tf_idf