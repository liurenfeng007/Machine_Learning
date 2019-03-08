# import torch.nn as nn
# import torch.nn.functional as F
# from base import BaseModel
#
#
# class MnistModel(BaseModel):
#     def __init__(self, num_classes=10):
#         super(MnistModel, self).__init__()
#         self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
#         self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
#         self.conv2_drop = nn.Dropout2d()
#         self.fc1 = nn.Linear(320, 50)
#         self.fc2 = nn.Linear(50, num_classes)
#
#     def forward(self, x):
#         x = F.relu(F.max_pool2d(self.conv1(x), 2))
#         x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
#         x = x.view(-1, 320)
#         x = F.relu(self.fc1(x))
#         x = F.dropout(x, training=self.training)
#         x = self.fc2(x)
#         return F.log_softmax(x, dim=1)

import torch.nn as nn
import torch.nn.functional as F
import torchtext.vocab as vocab
import torchtext
import torch


class CNN(nn.Module):
    def __init__(self, args, **kwargs):
        super(CNN, self).__init__(**kwargs)
        embed_num = args.embed_num  # 词表长21108
        embed_dim = args.embed_dim  # 嵌入维度
        text_field = args.text_field  # 词表
        channel_in = args.channel_in  # 输入信号的通道
        channel_out = args.kernel_num  # 输出信号的通道
        kernel_size = args.kernel_size  # 即filter的窗口大小
        class_num = args.class_num  # 词表类别数目
        # print(kernel_size)
        #print(embed_num)

        # embedding层
        self.embedding = nn.Embedding(num_embeddings=embed_num, embedding_dim=embed_dim)  # input (N*W) output (N*W*D)
        # 模型中制定与训练词向量
        self.embedding.weight.data.copy_(text_field.vocab.vectors)
        # 卷积层
        self.conv1 = nn.ModuleList([nn.Conv2d(in_channels=channel_in, out_channels=channel_out, kernel_size=(int(K), embed_dim)) for K in kernel_size])
        """
        nn.Conv2d(in_channels=channel_in, out_channels=channel_out, kernel_size=(K, embed_dim)) for K in kernel_size] # input (N,C_in,H_in,W_in) output (N,C_out,H_out,W_out)
        """
        self.dropout = nn.Dropout(args.dropout)
        self.fc = nn.Linear(len(kernel_size)*channel_out, class_num)

    def forward(self, x):
        x.permute(1, 0)
        # print(x.shape)
        x = self.embedding(x)  # output (N*W*D)
        # print(x.shape)
        x = x.unsqueeze(1)  # output (N*C_in*W*D)
        # print(x.shape)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.conv1]  # output [(N*C_out*W),(N*C_out*W),(N*C_out*W)]
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # output [(N*C_out),(N*C_out),(N*C_out)]
        x = torch.cat(x, 1)  # output (N*(C_out*3))
        x = self.dropout(x)  # output (N*(C_out*3))
        logit = self.fc(x)   # (N, C)
        return logit

