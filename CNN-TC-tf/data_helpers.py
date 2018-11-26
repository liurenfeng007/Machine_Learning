import tensorflow as tf
import numpy as np
import re

# tf.flags.DEFINE_string("positive_data_file", "./data/rt-polaritydata/rt-polarity.pos", "Data source for the positive data.")
# tf.flags.DEFINE_string("negative_data_file", "./data/rt-polaritydata/rt-polarity.neg", "Data source for the negative data.")
# FLAGS = tf.flags.FLAGS

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    """
    # [任意大小写字母数字(),!?'`]以外替换成空格
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    # 在各种缩写前面加空格
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

def load_data_and_labels(positive_data_file,negative_data_file):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    positive_examples = list(open(positive_data_file, "r", encoding='utf-8').readlines())
    positive_examples = [s.strip() for s in positive_examples ]
    negative_examples = list(open(negative_data_file, "r", encoding='utf-8').readlines())
    negative_examples = [s.strip() for s in negative_examples]
    # Split by words
    x_text = positive_examples + negative_examples
    x_text = [clean_str(sent) for sent in x_text]
    #print(x_text)
    # Generate labels
    positive_labels = [[0, 1] for _ in positive_examples]
    negative_labels = [[1, 0] for _ in negative_examples]
    y = np.concatenate([positive_labels,negative_labels], 0)
    #print(y)
    return [x_text,y]

# x_text, y = load_data_and_labels(FLAGS.positive_data_file, FLAGS.negative_data_file)

def batch_iter(data,batch_size,num_epochs,shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    # #计算每个epoch有多少个batch个数
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            # 函数shuffle与permutation都是对原来的数组进行重新洗牌（即随机打乱原来的元素顺序)
            shuffle_indices = np.random.permutation(np.arange(data_size))
            # 按照上面的乱序得到的新样本数据
            shuffle_data = data[shuffle_indices]
        else:
            shuffle_data = data
            # #生成batch
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            # 针对最后一个batch不足batch_size
            end_index = min((batch_num+1)*batch_size,data_size)
            yield shuffle_data[start_index:end_index]
            # yield：for循环执行时每次返回一个batch的data