from experiment import main

package = {}
trainf = "../Corpus/Reuters_train.txt"
testf = "../Corpus/Reuters_test.txt"


main(trainf,testf,algo="tf_idf",model = "svm")
#main(trainf,testf,algo="tf_dc",model="svm")
#main(trainf,testf,algo="tf_bdc",model="svm")
main(trainf,testf,algo="iqf_qf_icf",model="svm")
main(trainf,testf,algo="tf_rf",model="svm")
main(trainf,testf,algo="tf_chi",model="svm")
main(trainf,testf,algo="tf_eccd",model="svm")

# knn_neighbour 选用 1-35之间MicroF1即准确率最高的
#main(trainf,testf,algo="tf_idf",model = "knn",knn_neighbour=30)
# main(trainf,testf,algo="tf_dc",model="knn",knn_neighbour=5)
# main(trainf,testf,algo="tf_bdc",model="knn",knn_neighbour=1)
#main(trainf,testf,algo="iqf_qf_icf",model="knn",knn_neighbour=35)
#main(trainf,testf,algo="tf_rf",model="knn",knn_neighbour=10)
#main(trainf,testf,algo="tf_chi",model="knn",knn_neighbour=10)
#main(trainf,testf,algo="tf_eccd",model="knn",knn_neighbour=5)
#main(trainf,testf,algo="tf_ig",model="knn",knn_neighbour=10)