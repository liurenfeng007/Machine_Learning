from __future__ import print_function, division
import os
import codecs
from utils import zero_digits, IOB2, IOBES
import model

def load_dataset(path):
    sentence = []
    sentences = []


    for line in codecs.open(path, 'r', 'utf-8'):
        line = zero_digits(line.rstrip())
        if line:
                word = line.split()
                sentence.append(word)

        else:
            if len(sentence) > 0:
                if 'DOCSTART' not in sentence[0][0]:
                    sentences.append(sentence)
                sentence = []
    if len(sentence) > 0:
        if 'DOCSTART' not in sentence[0][0]:
            sentences.append(sentence)
    return sentences

def update_tag_scheme(sentences, tag_scheme):
    for i, s in enumerate(sentences):
        tags = [w[-1] for w in s]
        if tag_scheme == 'IOB2':
            # convert IOB to IOB2
            new_tags = IOB2(tags)
            for word, new_tag in zip(s, new_tags):
                word[-1] = new_tag
        elif tag_scheme == 'IOBES':
            new_tags = IOBES(tags)
            for word, new_tag in zip(s, new_tags):
                word[-1] = new_tag
        else:
            raise Exception('Unknown tagging scheme!')

def word_mapping(sentences):
    """
    Create a dictionary and a mapping of words, sorted by frequency.
    """
    words = [[w[0].lower() for w in s]for s in sentences]
    dict = {}
    for items in words:
        for item in items:
            if item not in dict:
                dict[item] = len(dict)
    dict['<PAD>'] = 10000001
    dict['<UNK>'] = 10000000
    print("Found %i unique words (%i in total)" % (
        len(dict), sum(len(x) for x in words)))
    return dict


def tag_mapping(sentences):
    """
    Create a dictionary and a mapping of tags, sorted by frequency.
    """
    tags = [[word[-1] for word in s] for s in sentences]
    dict = {}
    for items in tags:
        for item in items:
            if item not in dict:
                dict[item] = len(dict)
    dict[model.START_TAG] = -1
    dict[model.STOP_TAG] = -2
    print("Found %i unique named entity tags" % len(dict))
    return dict

def pre_dataset(sentences, word_dict, tag_dict):
    dataset = []
    for s in sentences[:100]:
        str_words = [w[0] for w in s]
        # print(str_words)
        words = [word_dict[w.lower()]
                 if w.lower() in word_dict else '<UNK>' for w in str_words]
        # print(words)
        tags = [tag_dict[w[-1]] for w in s]
        # print(tags)
        # print("/n")
        dataset.append([words, tags])
    return dataset


def prepare_batch(batch_size, dataset):
    index = 0
    batch_data = []

    def padding_data(data):
        max_len = max([len(s[0]) for s in data])
        for i in data:
            i.append(len(i[0]))
            i[0] = i[0] + (max_len - len(i[0])) * [17492]
            i[1] = i[1] + (max_len - len(i[1])) * [1]
        return data

    while True:
        if index + batch_size >= len(dataset):
            pad_data = padding_data(dataset[-batch_size:])
            batch_data.append(pad_data)
            break
        else:
            pad_data = padding_data(dataset[index:index + batch_size])
            index += batch_size
            batch_data.append(pad_data)
    return batch_data


def get_batch(batch_data):
    for data in batch_data:
        yield data


if __name__ == '__main__':
    train_data = 'F:/NER/BiLSTM-CRF/data/eng.train'
    sentences = load_dataset(train_data)
    print(sentences[:100])
    # update_tag_scheme(sentences, tag_scheme='IOBES')
    # print(sentences[:100])
    word_dict = word_mapping(sentences)
    print(word_dict)
    tag_dict = tag_mapping(sentences)
    print(tag_dict)
    dataset = pre_dataset(sentences, word_dict, tag_dict)
    print(dataset)
    print(len(dataset))
    batch_data = prepare_batch(2, dataset)
    print(batch_data[0])







