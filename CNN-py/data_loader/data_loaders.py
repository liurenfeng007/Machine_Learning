# from torchvision import datasets, transforms
# from base import BaseDataLoader
#
#
# class MnistDataLoader(BaseDataLoader):
#     """
#     MNIST data loading demo using BaseDataLoader
#     """
#     def __init__(self, data_dir, batch_size, shuffle, validation_split, num_workers, training=True):
#         trsfm = transforms.Compose([
#             transforms.ToTensor(),
#             transforms.Normalize((0.1307,), (0.3081,))
#             ])
#         self.data_dir = data_dir
#         self.dataset = datasets.MNIST(self.data_dir, train=training, download=True, transform=trsfm)
#         super(MnistDataLoader, self).__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

from torchtext import data
import re
import os
import random

class MRDataLoader(data.Dataset):
    def __init__(self, text_field, label_field, examples=None, **kwargs):

        def clean_str(string):
            """
            Tokenization/string cleaning for all datasets except for SST.
            Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
            """
            string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
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
            return string.strip()

        # self.train_batch_size =  train_batch_size()
        # self.eval_batch_size = eval_batch_size()

        # #define fields
        # text_field = data.Field(sequential=True, tokenize=tokenizer, lower=True)
        # label_field = data.Field(sequential=False, use_vocab=False)

        data_path = 'F:\python_study\pytorch-template-master\data'
        text_field.preprocessing = data.Pipeline(clean_str)
        fields = [('text', text_field), ('label', label_field)]
        if examples is None:
            examples = []
            with open(os.path.join(data_path +'/rt-polarity.neg'),errors='ignore') as f:
                examples += [data.Example.fromlist([line, 'negative'], fields)for line in f]
            # print(len(examples))# negative 5331
            with open(os.path.join(data_path, 'rt-polarity.pos'),errors='ignore') as f:
                examples += [data.Example.fromlist([line, 'positive'], fields)for line in f]
            # print(len(examples))# positive 5331
        super(MRDataLoader, self).__init__(examples, fields, **kwargs)

    @classmethod
    def splits(cls, text_field, label_field, dev_ratio=.1, shuffle=True, **kwargs):
        examples = cls(text_field, label_field, **kwargs).examples
        if shuffle:
            random.shuffle(examples)
        dev_index = -1 * int(dev_ratio * len(examples))
        # print(dev_index) # -1066
        return (cls(text_field, label_field, examples=examples[:dev_index]),
                cls(text_field, label_field, examples=examples[dev_index:]))




if __name__ == '__main__':
    text_field = data.Field(lower=True)
    label_field = data.Field(sequential=False)
    train_data, dev_data = MRDataLoader.splits(text_field, label_field)
    #print(train_data.shape)

    text_field.build_vocab(train_data, dev_data)
    label_field.build_vocab(train_data, dev_data)
    #x, y = load_data(batch_size=64)
    #print(x)
