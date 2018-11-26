import tensorflow as tf
import numpy as np


class TextCNN(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """
    """
    sequence_length 句子固定长度（不足补全，超过截断）
    num_classes 多分类, 分为几类.
    vocabulary_size 语料库的词典大小, 记为|D|.
    embedding_size 将词向量的维度, 由原始的 |D| 降维到 embedding_size.
    filter_size 卷积核尺寸
    num_filters 卷积核数量  
    l2_reg_lambda 正则化系数
    """
    def __init__(self,sequence_length,num_classes,vocab_size,embedding_size,filter_sizes,num_filters,l2_reg_lambda=0.0):

        # Placeholders for input, output and dropout
        # x*56 变量input_x存储句子矩阵，宽为sequence_length，长度自适应（=句子数量）
        # Tensor("input_x:0", shape=(?, 56), dtype=int32)
        self.input_x = tf.placeholder(tf.int32,[None,sequence_length],name="input_x")
        # x*2 input_y存储句子对应的分类结果，宽度为num_classes，长度自适应
        self.input_y = tf.placeholder(tf.float32,[None,num_classes],name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32,name="dropout_keep_prob")
        # Keeping track of L2 regulation loss (optional)
        # 变量dropout_keep_prob存储dropout参数，常量l2_loss为L2正则超参数
        l2_loss = tf.constant(0.0)

        """
        通过一个隐藏层, 将 one-hot 编码的词 投影 到一个低维空间中. 
        特征提取器，在指定维度中编码语义特征. 这样, 语义相近的词, 它们的欧氏距离或余弦距离也比较近.
        self.W可以理解为词向量词典，存储vocab_size个大小为embedding_size的词向量，随机初始化为-1~1之间的值；
        self.embedded_chars是输入input_x对应的词向量表示；size：[句子数量, sequence_length, embedding_size]
        self.embedded_chars_expanded是，将词向量表示扩充一个维度（embedded_chars * 1），维度变为[句子数量, sequence_length, embedding_size, 1]，方便进行卷积（tf.nn.conv2d的input参数为四维变量，见后文）
        函数tf.expand_dims(input, axis=None, name=None, dim=None)：在input第axis位置增加一个维度（dim用法等同于axis，官方文档已弃用）
        """
        # Embedding layer
        with tf.device('/cpu:0'),tf.name_scope("embedding"):
            self.W = tf.Variable(
                # 返回的n*k的词矩阵，产生于-1和1之间，产生的值是均匀分布的
                # vocab_size 18758 embedding_size 128
                tf.random_uniform([vocab_size,embedding_size],-1.0,1.0),
                name="W")
            self.embedded_chars = tf.nn.embedding_lookup(self.W,self.input_x)
            # 在最后一维位置增加一个维度
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars,-1)

        #  Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        for i,filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size,embedding_size,1,num_filters]
                # tf.truncated_normal函数产生正太分布，形状、均值和标准差自己设定
                W = tf.Variable(tf.truncated_normal(filter_shape,stddev=.01),name="W")
                b = tf.Variable(tf.constant(0.1,shape=[num_filters]),name="b")
                # input:输入的词向量，[句子数（图片数）batch, 句子定长（对应图高）,词向量维度（对应图宽）, 1（对应图像通道数）]
                # filter:卷积核，[卷积核的高度，词向量维度（卷积核的宽度），1（图像通道数），卷积核个数（输出通道数）]
                # strides:图像各维步长,一维向量，长度为4，图像通常为[1, x, x, 1]
                # padding:卷积方式，'SAME'为等长卷积, 'VALID'为窄卷积
                # 输出feature map：shape是[batch, height, width, channels]这种形式
                conv = tf.nn.conv2d(
                    self.embedded_chars_expanded,
                    W,
                    strides=[1,1,1,1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv,b),name="relu")
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1,sequence_length-filter_size+1,1,1],
                    strides=[1,1,1,1],
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(pooled)
                # value：待池化的四维张量，维度是[batch, height, width, channels]
                # ksize：池化窗口大小，长度（大于）等于4的数组，与value的维度对应，一般为[1,height,width,1]，batch和channels上不池化
                # strides:与卷积步长类似
                # padding：与卷积的padding参数类似
                # 返回值shape仍然是[batch, height, width, channels]这种形式
                # 池化后的结果append到pooled_outputs中。对每个卷积核重复上述操作，故pooled_outputs的数组长度应该为num_filters。

        # Combine all the pooled features
        # #将pooled_outputs中的值全部取出来然后reshape成[len(input_x),num_filters*len(filters_size)]，然后进行了dropout层防止过拟合
        num_filters_total = num_filters * len(filter_sizes)
        # tf.concat(values, concat_dim)连接values中的矩阵，concat_dim指定在哪一维（从0计数）连接。
        # values[i].shape = [D0, D1, ... Dconcat_dim(i), ...Dn]，连接后就是：[D0, D1, ... Rconcat_dim, ...Dn]
        self.h_pool = tf.concat(pooled_outputs, 3)
        # -1代表的含义是不用我们自己指定这一维的大小，函数会自动计算，但列表中只能存在一个-1
        self.h_pool_flat = tf.reshape(self.h_pool,[-1,num_filters_total])

        # Add dropout
        # tf.name_scope可以让变量有相同的命名，只是限于tf.Variable的变量
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat,self.dropout_keep_prob)

        # Final (unnormalized) scores and prediction
        with tf.name_scope("output"):
            W =tf.get_variable(
                "W",
                shape=[num_filters_total,num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1,shape=[num_classes]),name="b")
            l2_loss +=tf.nn.l2_loss(W)
            l2_loss +=tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.h_drop,W,b,name="scores")
            self.prediction = tf.argmax(self.scores,1,name="prediction")

        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores,labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_prediction = tf.equal(self.prediction,tf.argmax(self.input_y,1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction,"float"),name="accuracy")
