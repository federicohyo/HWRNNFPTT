# -*- coding: utf-8 -*-
"""train-lstm-1x128x10.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1OCdpoI-VLF7MUDSWAAc9hL8LayL9PNbv
"""



import numpy as np
import copy
import matplotlib.pyplot as plt
# import lstm as model
import lstm_new as model

# !python lstm.py

"""# Load & Preprocess Data"""

def LoadData(DirName,limit = 500):
    data = list();
    count = 0;
    with open(DirName) as f:
        for line in f:
            if count<limit:
                new_line = np.array(line.split(','))
                new_line = new_line.astype(np.float32)
                data.append(new_line)
                count += 1
    data = np.asarray(data)

    return data[:,0],data[:,1:]

NoTrain = 60000
NoTest = 10000
Y_train,x_train = LoadData("../data/mnist_train.csv", NoTrain)
Y_test,x_test = LoadData("../data/mnist_test.csv", NoTest)

print(f"Test set size: {x_test.shape[0]} x {x_test.shape[1]}")
print(f"Train set size: {x_train.shape[0]} x {x_train.shape[1]}")

#x_train_st = x_train#/255
#x_test_st = x_test#/255

# x_train_st[0]

x_train_st = (x_train-np.average(x_train))/np.std(x_train)
x_test_st = (x_test-np.average(x_train))/np.std(x_train)

# One-hot encoding train and test sets labels
y_train = np.zeros((Y_train.size, int(Y_train.max()) + 1))
# y_train = np.zeros((Y_train.size, 10))
y_train[np.arange(Y_train.size),Y_train.astype(int)] = 1.0;



print(f"Your decimal label is {Y_train[0]:.0f} and your one-hot encoded label is {y_train[0,:]}")
print(f"Correct decimal label is 5 and correct one-hot encoded label is [0. 0. 0. 0. 0. 1. 0. 0. 0. 0.]")

y_test  = np.zeros((Y_test.size, int(Y_test.max()) + 1))
# y_test  = np.zeros((Y_test.size, 10))
y_test[np.arange(Y_test.size),Y_test.astype(int)] = 1.0;

X = x_train_st.reshape(60000,784,1)
Y = y_train

print(X[0])

"""# Loss & Accuracy Definition"""

def cross_entropy(y_pred,y):
    """ Input Parameters: y_pred, y : array of float  Returns: c : float """

    # Compute loss
    c = ((np.log(abs(y_pred)))*y).sum(axis=1).sum()

    print(y_pred, y, c)
    return c

def nll_loss(y_pred, y):
    c = (-(y_pred)*y).sum(axis=1).sum()

    print(y_pred, y, c)
    return c


def labeling(x):
    # Set the label with the max probability to '1' and the rest to 0
    label = np.zeros((x.shape[0],Y.shape[1]))
    label[np.arange(x.shape[0]),x.argmax(axis=1)] = 1
    return label

def accuracy(y_pred,y):
    # Calculate the accuracy along the rows, averaging the results over the number of samples.
    acc = np.all(y_pred==y,axis=1).mean()
    return acc

"""# Instantiate Model"""

N_l = 128  # number of neurons in hidden layer

layers = np.array([1] +[N_l]+[Y.shape[1]]) #
print(layers)

np.random.seed(seed=0)

param = model.LstmParam(layers)

seq_len = 784
k = 28 # p
step = seq_len / k

print(X.shape)
sample = 1;  # index of the image we want to show

mean_train_loss_list = list()
train_acc_list = list()


# Initialize the batch size, the number of epochs, and the learning rate
n_samples = 60000 #X.shape[0]

batch_size = 100 
epochs = 60
lr = 0.01
alpha = 0.1

# Initialize the weights
# weights = init_weights(layers)
# w_g, w_lbd, w_rm = init_weights_derivatives(layers, weights)

# Q: predicted label distribution of last training epoch --> used to make auxiliary loss (oracle loss)
Q = np.zeros_like(Y)

target = np.zeros((batch_size, 10))


# Epoch for loop
for epoch in range(epochs):

    if (epoch==35) or (epoch==50):
        lr *= 0.1

    # Initialize the layers
    # ins, h, o = init_layer(layers,batch_size)
    # initiate hidden states for current batch
    Model = model.Lstm(seq_len, batch_size, layers, param)

    # Initialize the training loss and accuracy for each epoch
    train_loss = 0
    train_acc = 0

    # Create a random permutation for shuffling
    shuffle = np.random.permutation(n_samples)
    print(shuffle)


    # Shuffle dataset and create mini-batches for each epoch
    X_batches = np.array_split(X[shuffle],n_samples/batch_size)
    Y_batches = np.array_split(Y[shuffle],n_samples/batch_size)
    Q_batches = np.array_split(Q[shuffle],n_samples/batch_size)
    target = np.empty_like(Q_batches)

    # print(Y_batches)

    # Mini-batch for loop
    for b in range(int(n_samples/batch_size)):


        for p in range(28):

            cnt = p*28
            Model.state.forward(cnt, X_batches[b][:, cnt:(cnt+28), :] )

            # print(cnt, Model.state.Y_hat[cnt])
            # for v in (Model.state.Y_hat[783]):
            #     print(np.argmax(v))

            # print(Model.state.l1_h[cnt])
            # if(Model.state.l1_h[cnt][0].any()!=Model.state.l1_h[cnt][1].any()):
                # print("Diff", b, p)

            # print("batch, part", b, p)
            # print("weights:i", Model.param.l1_wi)
            # print(Model.state.l2_h[cnt], smax(Model.state.l2_h[cnt]), Model.state.Y_hat[cnt])

            # to make target for cross entropy and divergence term
            # if epoch == 0:
            #     beta = 1
            # else:
            beta = (p+1)/28

            target[b] = (Y_batches[b]) #+ (1-beta)*Q_batches[b]

            # print(cnt, target[b])
            # backwards - to calculate the gradients w.r.t. cross entropy loss and auxiliary loss
            Model.backward(batch_size, cnt+28-1, target[b], trunc_h=27, trunc_s=27)

            # print("l1", Model.param.l1_wi_diff[0:10,0:2])
            # print("l2", Model.param.l2_w_diff[0:10,0:2])
            # print(Model.param.l2_w[0:10,0:2])

            # update the weights by gradients obtained
            Model.param.apply_diff(beta*lr)
            # Model.param.apply_diff_fptt(lr,alpha)

            # print(Model.param.l1_wi[0:10,0:1])

        Q[shuffle[batch_size*b:batch_size*(b+1)]] = labeling(Model.state.Y_hat[seq_len-1]) #Q_batches[b]

        # train_loss += cross_entropy(Model.state.Y_hat[seq_len-1],Y_batches[b])
        train_loss += nll_loss(Model.state.Y_hat[seq_len-1], Y_batches[b])
        train_acc += accuracy(labeling(Model.state.Y_hat[seq_len-1]),Y_batches[b])
        print(epoch, b)

    mean_train_loss = train_loss/n_samples
    mean_train_loss_list.append(mean_train_loss)
    train_acc = (train_acc/len(X_batches))
    train_acc_list.append(train_acc)




    print(f"Epoch {epoch+1}: train_loss = {mean_train_loss:.3f} | train_acc = {train_acc:.3f} " )

