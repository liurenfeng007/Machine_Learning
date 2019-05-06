import torch
import torch.nn as nn
import torch.autograd as autograd
from torch.autograd import Variable
# a = [[[1,2,3,4],[1,2,3,4],3],[[4,5,6,8],[4,5,6,8],3],[[7,8,9,10],[7,8,9,10],3]]
#
# b,c,d = zip(*a)
# b = torch.LongTensor(b)
# print(b,c,d)
# print(b.shape)


# m = [[0,1,2,3,4],[3,5,4]]
# # n = [[5,6,7,8,9],[5,8,9]]
# #
# # for x in zip(m,n):
# #     print(x[0][0])
# #     print(x[1])
# #
# #     print(1)

input = autograd.Variable(torch.randn(128, 20))
print(input.size())