# import os
# import argparse
# import torch
# from tqdm import tqdm
# import data_loader.data_loaders as module_data
# import model.loss as module_loss
# import model.metric as module_metric
# import model.model as module_arch
# from train import get_instance
#
#
# def main(config, resume):
#     # setup data_loader instances
#     data_loader = getattr(module_data, config['data_loader']['type'])(
#         config['data_loader']['args']['data_dir'],
#         batch_size=512,
#         shuffle=False,
#         validation_split=0.0,
#         training=False,
#         num_workers=2
#     )
#
#     # build model architecture
#     model = get_instance(module_arch, 'arch', config)
#     model.summary()
#
#     # get function handles of loss and metrics
#     loss_fn = getattr(module_loss, config['loss'])
#     metric_fns = [getattr(module_metric, met) for met in config['metrics']]
#
#     # load state dict
#     checkpoint = torch.load(resume)
#     state_dict = checkpoint['state_dict']
#     if config['n_gpu'] > 1:
#         model = torch.nn.DataParallel(model)
#     model.load_state_dict(state_dict)
#
#     # prepare model for testing
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     model = model.to(device)
#     model.eval()
#
#     total_loss = 0.0
#     total_metrics = torch.zeros(len(metric_fns))
#
#     with torch.no_grad():
#         for i, (data, target) in enumerate(tqdm(data_loader)):
#             data, target = data.to(device), target.to(device)
#             output = model(data)
#             #
#             # save sample images, or do something with output here
#             #
#
#             # computing loss, metrics on test set
#             loss = loss_fn(output, target)
#             batch_size = data.shape[0]
#             total_loss += loss.item() * batch_size
#             for i, metric in enumerate(metric_fns):
#                 total_metrics[i] += metric(output, target) * batch_size
#
#     n_samples = len(data_loader.sampler)
#     log = {'loss': total_loss / n_samples}
#     log.update({met.__name__ : total_metrics[i].item() / n_samples for i, met in enumerate(metric_fns)})
#     print(log)
#
#
# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description='PyTorch Template')
#
#     parser.add_argument('-r', '--resume', default=None, type=str,
#                            help='path to latest checkpoint (default: None)')
#     parser.add_argument('-d', '--device', default=None, type=str,
#                            help='indices of GPUs to enable (default: all)')
#
#     args = parser.parse_args()
#
#     if args.resume:
#         config = torch.load(args.resume)['config']
#     if args.device:
#         os.environ["CUDA_VISIBLE_DEVICES"]=args.device
#
#     main(config, args.resume)

"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""
import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from utils.util import get_evaluation
import data_loader
import argparse
import shutil
import csv
from torchtext import data
from torchtext.vocab import Vectors
import numpy as np


def get_args():
    parser = argparse.ArgumentParser(
        """Implementation of the model described in the paper: Hierarchical Attention Networks for Document Classification""")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--data_path", type=str, default="data/test.csv")
    parser.add_argument("--pre_trained_model", type=str, default="trained_models/whole_model_cnn")
    parser.add_argument("--word2vec_path", type=str, default="data/glove.6B.50d.txt")
    parser.add_argument("--output", type=str, default="predictions")
    args = parser.parse_args()
    return args


def test(opt):
    test_params = {"batch_size": opt.batch_size,
                   "shuffle": False,
                   "drop_last": False}
    if os.path.isdir(opt.output):
        shutil.rmtree(opt.output)
    os.makedirs(opt.output)
    if torch.cuda.is_available():
        model = torch.load(opt.pre_trained_model)
    else:
        model = torch.load(opt.pre_trained_model, map_location=lambda storage, loc: storage)

    def load_data(opt, **kwargs):
        text_field = data.Field(lower=True)
        # NOTE 注意Field的参数设置，具体见代码
        label_field = data.Field(sequential=False, unk_token=None)
        # field可以共用，text和target即为绑定到example身上的属性
        train_data, dev_data = data_loader.MRDataLoader.splits(text_field, label_field, **kwargs)
        # 使用预训练的词向量进行训练
        # text_field.build_vocab(train, vectors="glove.6B.100d")
        vectors = Vectors(name='F:\python_study\pytorch-template-master\data\glove.6B.100d.txt')
        text_field.build_vocab(train_data, dev_data, vectors=vectors)
        label_field.build_vocab(train_data, dev_data, vectors=vectors)
        # NOTE 只能只用data.Iterator，不能用DataLoader
        train_iter, dev_iter = data.Iterator.splits(
            (train_data, dev_data), batch_sizes=(opt.batch_size, len(dev_data)), **kwargs)
        return train_iter, dev_iter  # train_loader=9596 dev_loader=1066


    test_iter, dev_iter = load_data(opt)

    if torch.cuda.is_available():
        model.cuda()
    model.eval()
    te_label_ls = []
    te_pred_ls = []
    for te_feature, te_label in test_iter:
        num_sample = len(te_label)
        if torch.cuda.is_available():
            te_feature = te_feature.cuda()
            te_label = te_label.cuda()
        with torch.no_grad():
            model._init_hidden_state(num_sample)
            te_predictions = model(te_feature)
            te_predictions = F.softmax(te_predictions)
        te_label_ls.extend(te_label.clone().cpu())
        te_pred_ls.append(te_predictions.clone().cpu())
    te_pred = torch.cat(te_pred_ls, 0).numpy()
    te_label = np.array(te_label_ls)

    fieldnames = ['True label', 'Predicted label', 'Content']
    with open(opt.output + os.sep + "predictions.csv", 'w') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames, quoting=csv.QUOTE_NONNUMERIC)
        writer.writeheader()
        for i, j, k in zip(te_label, te_pred, test_iter.texts):
            writer.writerow(
                {'True label': i + 1, 'Predicted label': np.argmax(j) + 1, 'Content': k})

    test_metrics = get_evaluation(te_label, te_pred,
                                  list_metrics=["accuracy", "loss", "confusion_matrix"])
    print("Prediction:\nLoss: {} Accuracy: {} \nConfusion matrix: \n{}".format(test_metrics["loss"],
                                                                               test_metrics["accuracy"],
                                                                               test_metrics["confusion_matrix"]))


if __name__ == "__main__":
    opt = get_args()
    test(opt)
