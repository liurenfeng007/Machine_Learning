import sys
sys.path.append('..')
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from os import walk
import matplotlib.pyplot as plt 

def neighbour(pathname, filename):
	global filedict
	file = open(pathname+"/"+filename).read().strip().split("\n")
	filedict[filename] = {}
	filedict[filename]["label"]=filename.replace("knn_","").replace(".txt","")
	filedict[filename]["x"] = []
	filedict[filename]["y"] = []
	for i in file:
		line = i.split("\t")
		#print line
		x = int(float(line[0]))
		y = (float(line[1]))
		print(x)
		print(y)
		filedict[filename]["x"].append(x)
		filedict[filename]["y"].append(y)

filedict = {}
for (dirpath,dirnames,filenames) in walk("./"):
	for file in filenames:
		if not file.endswith(".txt"):
			continue
		print(file)
		neighbour(dirpath, file)

for i in filedict:
	plt.plot(filedict[i]["x"],filedict[i]["y"],"x-",label =filedict[i]["label"] )

plt.legend(loc='lower right')
plt.show()