import sys
sys.path.append('..')
import os
import numpy as np
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# 获得corpus=[doucument1{},document2{},.......,,documentn{}]
# doucument={label:标签，document:标签+标签频次，split_sentence:一个标签下的所有词语，length:一个标签下的词语个数}
def preprocess(input, package, test=0):
    vocafreq = package["vocafreq"]
    voca = package["voca"]
    labelset = package["labelset"]
    input = np.array(open(input).read().split("\n"))
    corpus = []
    doccount = {}
    for i in input:
        document = {}
        sp = i.split("\t")
        label = sp[0]
        document["label"] = label
        if not label in doccount:
            doccount[label] = 0
        doccount[label] += 1
        docname = label+str(doccount[label])
        document["document"] = docname
        sp = sp[1].split(" ")
        while " " in sp:
            sp.remove(" ")
        while "" in sp:
            sp.remove("")
        document["split_sentence"] = sp
        if test==0:
            if label not in labelset:
                labelset.append(label)
            for word in sp:
                if not word in vocafreq:
                    vocafreq[word] = 0
                vocafreq[word] += 1
        document["length"] = len(sp)
        corpus.append(document)
    #训练集需要构建词汇表，并且需要进行维度压缩到5-2000
    if test == 0 :
        vocafreq = {x:vocafreq[x] for x in vocafreq if vocafreq[x]>5 and vocafreq[x]<2000}
        voca = list(vocafreq.keys())
        # ll = dict(Counter(vocafreq.values()))
        # plt.hist(ll.values())
        # plt.show()

    for i in corpus:
        i["split_sentence"] = [x for x in i["split_sentence"] if x in vocafreq]

    package["vocafreq"] = vocafreq
    package["voca"] = voca
    package["labelset"] = labelset
    return corpus
