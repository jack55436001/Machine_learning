#python -m visdom.server
# -*- coding: utf-8 -*-
import torch
from torch.autograd import Variable
import visdom
import numpy as np

vis = visdom.Visdom()
# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 64, 1000, 100, 10

# Create random Tensors to hold inputs and outputs, and wrap them in Variables.
x = Variable(torch.randn(N, D_in))
y = Variable(torch.randn(N, D_out), requires_grad=False)

# Use the nn package to define our model and loss function.
model = torch.nn.Sequential(
    torch.nn.Linear(D_in, H),
    torch.nn.ReLU(),
    torch.nn.Linear(H, D_out),
)
loss_fn = torch.nn.MSELoss(size_average=False)

# Use the optim package to define an Optimizer that will update the weights of
# the model for us. Here we will use Adam; the optim package contains many other
# optimization algoriths. The first argument to the Adam constructor tells the
# optimizer which Variables it should update.
learning_rate = 1e-2
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
learn_t = 500
patch = 10


X = np.zeros(patch)
Y = np.zeros(patch)

first = False
win  = None
for t in range(learn_t):
    # Forward pass: compute predicted y by passing x to the model.
    y_pred = model(x)
    
    # Compute and print loss.
    loss = loss_fn(y_pred, y)

    #0~9 10~19 
    X[t%10] = t
    Y[t%10] = loss.data[0]
    if first==False:
       win = vis.line(
        Y = Y,
        X = X,
    	opts=dict(markers=False),
       )
       first = True

    if t%10 == 9:
       if first==False:
          win = vis.line(
          Y = Y,
          X = X,
    	  opts=dict(markers=False),
          )
          first = True
       else:
          vis.line(
          X=X,
          Y=Y,
          win=win,
          update='append'
          )
    # Before the backward pass, use the optimizer object to zero all of the
    # gradients for the variables it will update (which are the learnable weights
    # of the model)
    optimizer.zero_grad()

    # Backward pass: compute gradient of the loss with respect to model
    # parameters
    loss.backward()

    # Calling the step function on an Optimizer makes an update to its
    # parameters
    optimizer.step()



