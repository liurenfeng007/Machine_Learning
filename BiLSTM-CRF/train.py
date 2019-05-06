from __future__ import print_function
import torch
import torch.nn as nn
from torch.autograd import Variable
import pickle
import numpy as np
import codecs
import argparse
import datahelper
from CRF import BiLSTM_CRF


def get_args():
    parser = argparse.ArgumentParser(
        """Implementation of the model described in the paper: Hierarchical Attention Networks for Document Classification""")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--embed_dim", type=int, default=50)
    parser.add_argument("--num_epoches", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.01)# learning rate
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--tag_scheme", type=str, default='IOBES', help="Tagging scheme (IOB or IOBES)")
    parser.add_argument("--hidden_dim", type=int, default=400)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--train_set", type=str, default="data/eng.train")
    parser.add_argument("--dev_set", type=str, default="data/eng.testa")
    parser.add_argument("--test_set", type=str, default="data/eng.testb")
    parser.add_argument("--embed_type", type=int, default=1, help="need pretrained embeddding, 0 is need")
    parser.add_argument("--word2vec_path", type=str, default="data/glove.6B.50d.txt")
    parser.add_argument("--log_path", type=str, default="tensorboard/han_voc")
    parser.add_argument("--mapping_file", type=str, default="models/mapping.pkl")
    parser.add_argument("--saved_path", type=str, default="trained_models")
    args = parser.parse_args()
    return args

def train(opt):

    if torch.cuda.is_available():
        torch.cuda.manual_seed(123)
    else:
        torch.manual_seed(123)  # 同时设置cpu和gpu的随机种子
    train_sentences = datahelper.load_dataset(opt.train_set)
    datahelper.update_tag_scheme(train_sentences, opt.tag_scheme)
    word_dict = datahelper.word_mapping(train_sentences)
    tag_dict = datahelper.tag_mapping(train_sentences)
    train_data = datahelper.pre_dataset(train_sentences, word_dict, tag_dict)
    batch_data = datahelper.prepare_batch(opt.batch_size,train_data)

    dev_sentences = datahelper.load_dataset(opt.dev_set)
    datahelper.update_tag_scheme(dev_sentences, opt.tag_scheme)
    dev_word_dict = datahelper.word_mapping(dev_sentences)
    dev_tag_dict = datahelper.tag_mapping(dev_sentences)
    dev_data = datahelper.pre_dataset(dev_sentences, dev_word_dict, dev_tag_dict)
    batch_dev_data = datahelper.prepare_batch(opt.batch_size, dev_data)



    word_embeds = np.random.uniform(-np.sqrt(0.06), np.sqrt(0.06), (len(word_dict), opt.embed_dim))
    if opt.embed_type == 0:
        pre_word_embeds = {}
        for i, line in enumerate(codecs.open(opt.word2vec_path, 'r', 'utf-8')):
            s = line.strip().split()
            if len(s) == opt.embed_dim + 1:
                pre_word_embeds[s[0]] = np.array([float(i) for i in s[1:]])


        for w in word_dict:
            if w in pre_word_embeds:
                word_embeds[word_dict[w]] = pre_word_embeds[w]
            elif w.lower() in pre_word_embeds:
                word_embeds[word_dict[w]] = pre_word_embeds[w.lower()]



    model = BiLSTM_CRF(vocab_size=len(word_dict),
                       embedding_dim=opt.embed_dim,
                       hidden_dim=opt.hidden_dim,
                       tag_dict=tag_dict,
                       batch_size=opt.batch_size,
                       dropout=opt.dropout,
                       pre_word_embeds=word_embeds)
    # optimizer = torch.optim.SGD(model.parameters(), lr=opt.lr, momentum=opt.momentum)
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)

    def evaluate(batch_dev):
        sentences, labels, length = zip(*batch_dev)
        scores, paths = model(sentences,length)
        # print(dev_data)
        # real_sentences, real_labels = zip(*dev_data)
        # print(real_labels)
        # print(labels)
        # print(paths)


        right = 0.
        found = 0.
        for tar in zip(paths, labels, length):
            #
            # print(tar[0])
            # print(tar[1][:tar[2]])
            for i in range(tar[2]):
                if tar[0][i] == tar[1][i]:
                    right += 1
            found += tar[2]
            print(right)
            print(found)
        return right, found

    for epoch in range(opt.num_epoches):
        acc = 0
        index = 0
        for batch in batch_data:
            index += 1
            model.zero_grad()
            sentences, tags, length = zip(*batch)
            sentence_in = torch.LongTensor(sentences)
            # print(sentence_in.size())
            targets = torch.LongTensor(tags)
            length = torch.LongTensor(length)

            if torch.cuda.is_available():
                loss = model.neg_log_likelihood(sentence_in.cuda(), targets.cuda())
            else:
                loss = model.neg_log_likelihood(sentence_in, targets, length)
            loss.backward()
            # torch.nn.utils.clip_grad_norm(model.parameters(), 5.0)
            optimizer.step()
            torch.save(model.state_dict(), 'params.pkl')
            # scores, paths = model(sentences)
            # print(tags)
            # print(paths)
            print("Epoch: {}/{}, Iteration: {}/{}, Lr: {}, Loss: {}".format(
                epoch + 1,
                opt.num_epoches,
                index,
                len(batch_data),
                optimizer.param_groups[0]['lr'],
                loss))
        for batch_dev in batch_dev_data:
            rights = 0
            founds = 0
            right, found = evaluate(batch_dev)
            rights += right
            founds += found
            acc = rights/found
        print("Epoch: {}/{},Accuracy: {}".format(
                epoch + 1,
                opt.num_epoches,acc))








if __name__ == '__main__':
    opt = get_args()
    train(opt)



