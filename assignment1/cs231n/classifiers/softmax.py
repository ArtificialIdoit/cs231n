import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  #注意是softmax而不是sigmoid！
  num_train = X.shape[0]
  num_class = y.shape
  for i in range(X.shape[0]):
    score = X[i,:].dot(W)
    score = score - score.max()
    softmax = np.exp(score)
    softmax = softmax/softmax.sum()
    loss += -np.log(softmax[y[i]])
    softmax[y[i]] -= 1
    dW += X[i,:].reshape((W.shape[0],1))*softmax.reshape((1,W.shape[1]))
  loss/= num_train
  # loss也要有正则化项！！！
  loss += reg*np.square(W).sum()
  dW/=num_train
  dW += 2*reg*W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train = X.shape[0]
  num_class = y.shape
  score = X.dot(W)
  score = score - score.max(axis = 1).reshape(num_train,1)
  softmax = np.exp(score)
  softmax = softmax/softmax.sum(axis =1).reshape(num_train,1)
  loss += -np.log(softmax[np.arange(num_train),y]).sum()
  #softmax没必要都求吧？
  #有必要，求dW要用到
  softmax[np.arange(num_train),y] -= 1
  dW += X.T.dot(softmax)
  loss/= num_train
  loss += reg*np.square(W).sum()
  dW/=num_train
  dW += 2*reg*W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

