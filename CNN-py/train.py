# import os
# import json
# import argparse
# import torch
# import data_loader.data_loaders as module_data
# import model.loss as module_loss
# import model.metric as module_metric
# import model.model as module_arch
# from trainer import Trainer
# from utils import Logger
#
#
# def get_instance(module, name, config, *args):
#     return getattr(module, config[name]['type'])(*args, **config[name]['args'])
#
# def main(config, resume):
#     train_logger = Logger()
#
#     # setup data_loader instances
#     data_loader = get_instance(module_data, 'data_loader', config)
#     valid_data_loader = data_loader.split_validation()
#
#     # build model architecture
#     model = get_instance(module_arch, 'arch', config)
#     print(model)
#
#     # get function handles of loss and metrics
#     loss = getattr(module_loss, config['loss'])
#     metrics = [getattr(module_metric, met) for met in config['metrics']]
#
#     # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
#     trainable_params = filter(lambda p: p.requires_grad, model.parameters())
#     optimizer = get_instance(torch.optim, 'optimizer', config, trainable_params)
#     lr_scheduler = get_instance(torch.optim.lr_scheduler, 'lr_scheduler', config, optimizer)
#
#     trainer = Trainer(model, loss, metrics, optimizer,
#                       resume=resume,
#                       config=config,
#                       data_loader=data_loader,
#                       valid_data_loader=valid_data_loader,
#                       lr_scheduler=lr_scheduler,
#                       train_logger=train_logger)
#
#     trainer.train()
#
# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description='PyTorch Template')
#     parser.add_argument('-c', '--config', default=None, type=str,
#                            help='config file path (default: None)')
#     parser.add_argument('-r', '--resume', default=None, type=str,
#                            help='path to latest checkpoint (default: None)')
#     parser.add_argument('-d', '--device', default=None, type=str,
#                            help='indices of GPUs to enable (default: all)')
#     args = parser.parse_args()
#
#     if args.config:
#         # load config file
#         config = json.load(open(args.config))
#         path = os.path.join(config['trainer']['save_dir'], config['name'])
#     elif args.resume:
#         # load config file from checkpoint, in case new config file is not given.
#         # Use '--config' and '--resume' arguments together to load trained model and train more with changed config.
#         config = torch.load(args.resume)['config']
#     else:
#         raise AssertionError("Configuration file need to be specified. Add '-c config.json', for example.")
#
#     if args.device:
#         os.environ["CUDA_VISIBLE_DEVICES"] = args.device
#
#     main(config, args.resume)

import os
import numpy as np
import argparse
import shutil
import torch
import torch.nn.functional as F
from torchtext import data
from torchtext.vocab import Vectors
from model.model import CNN
from utils.util import get_evaluation
import data_loader
from tensorboardX import SummaryWriter

def get_argas():
    parser = parser = argparse.ArgumentParser(
        """CNN For Text""")
    # learning
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_epoches", type=int, default=8, help='number of epochs for train')
    parser.add_argument("--lr", type=float, default=0.1, help="learning rate")
    parser.add_argument("--save_path", type=str, default="trained_models", help="where to save the snapshot")
    parser.add_argument("--data_path", type=str, default="F:\python_study\pytorch-template-master\data")
    parser.add_argument("--word2vec_path", type=str, default="F:\python_study\pytorch-template-master\data\glove.6B.100d.txt")
    parser.add_argument("--log_path", type=str, default="tensorboard/CNN", help="where to save the snapshot")
    parser.add_argument('-test-interval', type=int, default=100,
                        help='how many steps to wait before testing')
    parser.add_argument("--es_min_delta", type=float, default=0.0,
                        help="Early stopping's parameter: minimum change loss to qualify as an improvement")
    parser.add_argument("--es_patience", type=int, default=5,
                        help="Early stopping's parameter: number of epochs with no improvement after which training will be stopped. Set to 0 to disable this technique.")

    # model
    parser.add_argument("--channel_in", type=int, default=1, help="channel to input")
    parser.add_argument("--dropout", type=float, default=0.5, help="the probability for dropout")
    parser.add_argument('-kernel-size', type=str, default='345',
                        help='comma-separated kernel size to use for convolution')
    parser.add_argument('--kernel-num', type=int, default=100, help='number of each kind of kernel')
    parser.add_argument('--norm', type=int, default=3, help='l2 constraint of parameters')
    parser.add_argument('-embed-dim', type=int, default=100, help='number of embedding dimension ')
    parser.add_argument('-static', action='store_true', default=False, help='fix the embedding')

    # data
    parser.add_argument('-shuffle', action='store_true', default=True, help='shuffle the data every epoch')

    args = parser.parse_args()
    return args




def train(opt):
    if torch.cuda.is_available():
        torch.cuda.manual_seed(123)
    else:
        torch.manual_seed(123)
    output_file = open(opt.save_path + "_log.text", "w")
    output_file.write("Model's parameters: {}".format(vars(opt)))

    text_field = data.Field(lower=True)
    # NOTE 注意Field的参数设置，具体见代码
    label_field = data.Field(sequential=False, unk_token=None)
    # field可以共用，text和target即为绑定到example身上的属性
    train_data, dev_data = data_loader.MRDataLoader.splits(text_field, label_field)
    # 使用预训练的词向量进行训练
    # text_field.build_vocab(train, vectors="glove.6B.100d")
    vectors = Vectors(name=opt.word2vec_path)
    text_field.build_vocab(train_data, dev_data, vectors=vectors)
    label_field.build_vocab(train_data, dev_data, vectors=vectors)
    # NOTE 只能只用data.Iterator，不能用DataLoader
    train_iter, dev_iter = data.Iterator.splits(
        (train_data, dev_data), sort_key=lambda x: len(x.text), batch_sizes=(opt.batch_size, len(dev_data)))
    # train_loader=9596 dev_loader=1066
    # print(len(dev_data))



    opt.embed_num = len(text_field.vocab)
    opt.class_num = len(label_field.vocab)
    opt.text_field = text_field
    # print(text_field.vocab.vectors.size())
    model = CNN(opt)

    if os.path.isdir(opt.log_path):
        shutil.rmtree(opt.log_path)
    os.makedirs(opt.log_path)
    writer = SummaryWriter(opt.log_path)

    if torch.cuda.is_available():
        model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)

    best_loss = 1e5
    best_epoch = 0
    model.train()
    num_iterator = len(train_iter)
    # print(num_iterator)
    num_dev_iterator = len(dev_iter)
    for epoch in range(opt.num_epoches):
        iter = 0
        for batch in train_iter:
            feature = batch.text.permute(1, 0)
            # print(feature.shape)
            label = batch.label
            if torch.cuda.is_available():
                feature = feature.cuda()
                label = label.cuda()

            optimizer.zero_grad()
            predictions = model(feature)
            loss = F.cross_entropy(predictions, label)
            loss.backward()
            optimizer.step()
            training_metrics = get_evaluation(label.cpu().numpy(), predictions.cpu().detach().numpy(),
                                              list_metrics=["accuracy"])
            print("Epoch: {}/{}, Iteration: {}/{}, Lr: {}, Loss: {}, Accuracy: {}".format(
                epoch + 1,
                opt.num_epoches,
                iter + 1,
                num_iterator,
                optimizer.param_groups[0]['lr'],
                loss, training_metrics["accuracy"]))
            iter = iter + 1
            writer.add_scalar('Train/Loss', loss, epoch * num_iterator + iter)
            writer.add_scalar('Train/Accuracy', training_metrics["accuracy"], epoch * num_iterator + iter)

            if epoch % opt.test_interval == 0:
                model.eval()
                loss_ls = []
                te_label_ls = []
                te_pred_ls = []
                for dev_batch in dev_iter:
                    te_label = dev_batch.label
                    te_feature = dev_batch.text.permute(1, 0)
                    num_sample = len(te_label)

                    if torch.cuda.is_available():
                        te_feature = te_feature.cuda()
                        te_label = te_label.cuda()

                    with torch.no_grad():
                        te_predictions = model(te_feature)
                    te_loss = F.cross_entropy(te_predictions, te_label)
                    loss_ls.append(te_loss * num_sample)
                    te_label_ls.extend(te_label.clone().cpu())
                    te_pred_ls.append(te_predictions.clone().cpu())
                te_loss = sum(loss_ls) / num_dev_iterator
                te_pred = torch.cat(te_pred_ls, 0)
                te_label = np.array(te_label_ls)
                test_metrics = get_evaluation(te_label, te_pred.numpy(), list_metrics=["accuracy", "confusion_matrix"])
                output_file.write(
                    "Epoch: {}/{} \nTest loss: {} Test accuracy: {} \nTest confusion matrix: \n{}\n\n".format(
                        epoch + 1, opt.num_epoches,
                        te_loss,
                        test_metrics["accuracy"],
                        test_metrics["confusion_matrix"]))
                print("Epoch: {}/{}, Lr: {}, Loss: {}, Accuracy: {}".format(
                    epoch + 1,
                    opt.num_epoches,
                    optimizer.param_groups[0]['lr'],
                    te_loss, test_metrics["accuracy"]))

                writer.add_scalar('Test/Loss', te_loss, epoch)
                writer.add_scalar('Test/Accuracy', test_metrics["accuracy"], epoch)

                model.train()
                if te_loss + opt.es_min_delta < best_loss:
                    best_loss = te_loss
                    best_epoch = epoch
                    torch.save(model, os.path.join(opt.save_path + "whole_model_cnn"))

                # Early stopping
                if epoch - best_epoch > opt.es_patience > 0:
                    print("Stop training at epoch {}. The lowest loss achieved is {}".format(epoch, te_loss))
                    break

if __name__ == '__main__':
    opt = get_argas()
    train(opt)