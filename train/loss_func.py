import torch
import torch.nn as nn

torch.manual_seed(42)
# for i in range(50):
input = torch.randn(10,3, requires_grad=True)
# target = torch.empty(3, dtype=torch.long).random_(3)

# The Count  of target and the input must be the same.
target=torch.empty(10,dtype=torch.long).random_(3)

cross_entropy_loss = nn.CrossEntropyLoss() # need to target be Long
output = cross_entropy_loss(input, target)
# output.backward()
# output.backward(retain_graph=True)
# print(output)

#------------------------------------------------------------------------------------------------
# Hinge_loss used for binary classification and also used for unsupervised learning. 
import torch
import torch.nn as nn

input = torch.randn(4, 5, requires_grad=True)
target = torch.randn(4, 5)

hinge_loss = nn.HingeEmbeddingLoss()
output = hinge_loss(input, target)
output.backward()

# print('input: ', input)
# print('target: ', target)
# print('output: ', output)

#------------------------------------------------------------------------------------------------
import torch
import torch.nn as nn

input_one = torch.randn(5, requires_grad=True)
input_two = torch.randn(5, requires_grad=True)
target = torch.randn(5).sign()

ranking_loss = nn.MarginRankingLoss()
output = ranking_loss(input_one, input_two, target)
output.backward()

# print('input one: ', input_one)
# print('input two: ', input_two)
# print('target: ', target)

#------------------------------------------------------------------------------------------------
# When could it be used?
# Approximating complex functions
# Multi-class classification tasks
# If you want to make sure that the distribution of predictions is similar to that of training data

input = torch.randn(2, 3, requires_grad=True)
target = torch.randn(2, 3)
kl_loss = nn.KLDivLoss(reduction = 'batchmean')
output = kl_loss(input, target)
output.backward()

# print('input: ', input)
# print('target: ', target)
# print('output: ', output)