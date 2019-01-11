import numpy as np

def L_I(x, y, W):
    delta = 1.0 #set delta initially as one differnce
    scores = W.dot(x) #score is the dot product with x
    correct_class_score = scores[y] #correct class is score of the correct class
    D = W.shape[0] #Take the shape of W, only the number of classifiers
    loss_i = 0.0 #loss for i
    for j in range(D): #take each value in range
        if j == y: #If this is the same value we don't calculate loss
            continue
        loss_i += max(0, scores[j] - correect_class_score + dela) #add difference
    return loss_i #return the whole lass

def L_i_vectorized(x, y, W):
    delta = 1.0 #delta requires difference of one
    scores = W.dot(x) #calculate the scores
    margins = np.max(0, scores - scores[y] + delta) #the margin is the same
    margins[y] = 0 #add the margin for the non-error values
    loss_i = np.sum(margins) #total loss is sum of margins
    return loss_i #return loss

def L(x, y, W):
  """
  fully-vectorized implementation :
  - X holds all the training examples as columns (e.g. 3073 x 50,000 in CIFAR-10)
  - y is array of integers specifying correct class (e.g. 50,000-D array)
  - W are weights (e.g. 10 x 3073)
  """
  # evaluate loss over all examples in X without using any for loops
  # left as exercise to reader in the assignment
  delta = 1.0
  scores = W.dot(x)
  margins = np.max(0, scores - scores[y] + delta)
  margins[y] = 0
  r_pen = W[x, y]**2
  loss_i = np.sum(margins) + np.sum(r_pen)
  return loss_i
