"""
Minimal character-level LSTM model. Written by Ngoc Quan Pham
Code structure borrowed from the Vanilla RNN model from Andreij Karparthy @karparthy.
BSD License
"""
import numpy as np
from random import uniform
import sys

from numpy.core.fromnumeric import shape


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def dsigmoid(y):
    return y * (1 - y)


def dtanh(x):
    return 1 - x * x


# The numerically stable softmax implementation
def softmax(x):
    # assuming x shape is [feature_size, batch_size]
    e_x = np.exp(x - np.max(x, axis=0))
    return e_x / e_x.sum(axis=0)


# data I/O
data = open('data/whatsapp.txt', 'r').read()  # should be simple plain text file
chars = sorted(list(set(data)))
data_size, vocab_size = len(data), len(chars)
print('data has %d characters, %d unique.' % (data_size, vocab_size))
char_to_ix = {ch: i for i, ch in enumerate(chars)}
ix_to_char = {i: ch for i, ch in enumerate(chars)}
std = 0.1

option = sys.argv[1]

# hyperparameters
emb_size = 12
hidden_size = 16  # size of hidden layer of neurons
seq_length = 32  # number of steps to unroll the RNN for
learning_rate = 0.02
max_updates = 500000
batch_size = 8

concat_size = emb_size + hidden_size

# model parameters
# char embedding parameters
Wex = np.random.randn(emb_size, vocab_size) * std  # embedding layer

# LSTM parameters
Wf = np.random.randn(hidden_size, concat_size) * std  # forget gate
Wi = np.random.randn(hidden_size, concat_size) * std  # input gate
Wo = np.random.randn(hidden_size, concat_size) * std  # output gate
Wc = np.random.randn(hidden_size, concat_size) * std  # c term

bf = np.zeros((hidden_size, 1))  # forget bias
bi = np.zeros((hidden_size, 1))  # input bias
bo = np.zeros((hidden_size, 1))  # output bias
bc = np.zeros((hidden_size, 1))  # memory bias

# Output layer parameters
Why = np.random.randn(vocab_size, hidden_size) * std  # hidden to output
by = np.random.randn(vocab_size, 1) * std  # output bias

data_stream = np.asarray([char_to_ix[char] for char in data])

bound = (data_stream.shape[0] // (seq_length * batch_size)) * (seq_length * batch_size)
cut_stream = data_stream[:bound]
cut_stream = np.reshape(cut_stream, (batch_size, -1))


def forward(inputs, targets, memory):
    """
    inputs,targets are both list of integers.
    hprev is Hx1 array of initial hidden state
    returns the loss, gradients on model parameters, and last hidden state
    """
    hprev, cprev = memory

    # xs: inputs
    # wes: Word embeddings at timestamp
    # zs: concatenated input and h
    # fs: forget_states
    # ins: input gate state at timestamp
    # cs: cell state
    # c_t: candidate content at timestamp
    # os: output gate
    # hs: hidden
    # outputs
    # ps: softmax output
    # ls: label as one hot vector
    # yt: output of lstm cell (without softmax)


    xs, wes, hs, yt, ps, cs, zs, ins, c_t, ls = {}, {}, {}, {}, {}, {}, {}, {}, {}, {}
    os, fs = {}, {}

    #old hidden state 
    hs[-1] = np.copy(hprev)

    #old cell state
    cs[-1] = np.copy(cprev)

    loss = 0
    input_length = inputs.shape[0]

    # forward pass
    for t in range(input_length):
        xs[t] = np.zeros((vocab_size, batch_size))  # encode in 1-of-k representation
        for b in range(batch_size):
            xs[t][inputs[t][b]][b] = 1

        # convert word indices to word embeddings
        wes[t] = np.dot(Wex, xs[t])

        # LSTM cell operation
        # first concatenate the input and h to get z        
        zs[t] = np.row_stack((hs[t - 1], wes[t]))
        # compute the forget gate
        # f = sigmoid(Wf * z + bf)
        fs[t] = sigmoid(Wf.dot(zs[t])+bf)

        # compute the input gate
        # i = sigmoid(Wi * z + bi)
        ins[t] = sigmoid(Wi.dot(zs[t])+bi)

        # output gate
        #o = sigmoid(Wo * z + bo)
        os[t] = sigmoid(Wo.dot(zs[t]) + bo)

        # compute the candidate memory
        # c_ = tanh(Wc * z + bc)
        c_t[t] = np.tanh(Wc.dot(zs[t])+bc)


        # new memory: applying forget gate on the previous memory
        # and then adding the input gate on the candidate memory
        # c_t = f * c_(t-1) + i * c_
        cs[t] = (fs[t] * cs[t-1]) + (ins[t] * c_t[t])
       
        #new hidden state
        hs[t] = os[t] * np.tanh(cs[t])

        # DONE LSTM
        # output layer - softmax and cross-entropy loss
        # unnormalized log probabilities for next chars
        # softmax for probabilities for next chars

        #predicted word for timestamp t
        yt[t] = Why.dot(hs[t])+by
        ps[t] = softmax(yt[t])

        # label
        ls[t] = np.zeros((vocab_size, batch_size))
        for b in range(batch_size):
            ls[t][targets[t][b]][b] = 1

        # cross-entropy loss
        loss_t = np.sum(-np.log(ps[t]) * ls[t])
        loss += loss_t
        #loss += -np.log(ps[t][targets[t],0])

    activations = (xs, wes, zs,fs, ins, cs, c_t, os, hs, ps, ls)
    memory = (hs[input_length - 1], cs[input_length -1])

    return loss, activations, memory


def backward(activations, clipping=True):
    xs,wes,zs,fs,ins,cs,c_t,os,hs,ps,ls = activations

    # backward pass: compute gradients going backwards
    # Here we allocate memory for the gradients
    dWex, dWhy = np.zeros_like(Wex), np.zeros_like(Why)
    dby = np.zeros_like(by)
    dWf, dWi, dWc, dWo = np.zeros_like(Wf), np.zeros_like(Wi), np.zeros_like(Wc), np.zeros_like(Wo)
    dbf, dbi, dbc, dbo = np.zeros_like(bf), np.zeros_like(bi), np.zeros_like(bc), np.zeros_like(bo)

    dhnext = np.zeros_like(hs[0])
    dcnext = np.zeros_like(cs[0])

    input_length = len(xs)

    # back propagation through time starts here
    for t in reversed(range(input_length)):
        # computing the gradients here

        # Softmax loss gradient from helpsheet P - y
        dy = ps[t]-ls[t] 

        #(Fully Conected) Output Layer
        dWhy += dy.dot(hs[t].T)
        # bias gradient (row sum)
        dby += dy.sum(axis=-1,keepdims=True)

        # hidden state gradient, add dhnext because it will be used for stacking in next timestamp [h,input]
        dhs = np.dot(Why.T, dy) + dhnext

        #Gradient cell state
        dcs = dhs * dtanh(cs[t])* os[t] + dcnext
        # Gradient for c_old in cs = fs * cs[t-1] + ins * c_t
        dcnext = fs[t] * dcs 


        #Output Gate
        dos = np.tanh(cs[t]) * dhs
        dos = dsigmoid(os[t]) * dos
        dWo += dos.dot(zs[t].T)
        dbo += dos.sum(axis=1, keepdims=True)
        dXo = Wo.T.dot(dos)

      

        #Forget Gate Gradient
        dfs = dsigmoid(fs[t]) * (cs[t-1] * dcs)
        dWf += dfs.dot(zs[t].T)
        dbf += dfs.sum(axis=-1,keepdims=True)
        dXf = Wf.T.dot(dfs)

        #Input Gate Gradient
        dins = (c_t[t] * dcs)
        dins_pre = dsigmoid(ins[t]) * dins
        dWi += dins_pre.dot(zs[t].T)
        dbi += dins_pre.sum(axis=-1, keepdims=True)
        dXi = Wi.T.dot(dins_pre)

        #Candidate Content Gradient
        dc_t = dtanh(c_t[t]) * (ins[t] * dcs)
        dWc += dc_t.dot(zs[t].T)
        dbc += dc_t.sum(axis=-1, keepdims=True)
        dXc = Wc.T.dot(dc_t)

        # As zs was used in multiple gates, the gradient must be accumulated here
        dZs = dXo + dXc + dXi + dXf

        # Split the concatenated Zs, so that we get our gradient of h_old
        dhnext = dZs[:hidden_size,:]

        # Split the concatenated Zs, for embedding
        dwes = dZs[hidden_size:,:]
        dWex += dwes.dot(xs[t].T)

       


    # clip to mitigate exploding gradients
    if clipping:
        for dparam in [dWex, dWf, dWi, dWo, dWc, dbf, dbi, dbo, dbc, dWhy, dby]:
            np.clip(dparam, -5, 5, out=dparam)

    gradients = (dWex, dWf, dWi, dWo, dWc, dbf, dbi, dbo, dbc, dWhy, dby)

    return gradients


def sample(memory, seed_ix, n):
    """
    sample a sequence of integers from the model
    h is memory state, seed_ix is seed letter for first time step
    """
    h, c = memory
    x = np.zeros((vocab_size, 1))
    x[seed_ix] = 1
    ixes = []

    for t in range(n):
          # convert word indices to word embeddings
        wes = np.dot(Wex, x)

        # LSTM cell operation
        # first concatenate the input and h to get z
        zs = np.row_stack((h, wes))

        # compute the forget gate
        # f = sigmoid(Wf * z + bf)
        fs = sigmoid(Wf.dot(zs)+bf)

        # compute the input gate
        # i = sigmoid(Wi * z + bi)
        ins = sigmoid(Wi.dot(zs)+bi)

        # compute the candidate memory
        # c_ = tanh(Wc * z + bc)
        c_t = np.tanh(Wc.dot(zs)+bc)


        # new memory: applying forget gate on the previous memory
        # and then adding the input gate on the candidate memory
        # c_t = f * c_(t-1) + i * c_
        c = fs * c + ins * c_t

        # output gate
        #o = sigmoid(Wo * z + bo)
        os = sigmoid(Wo.dot(zs) +bo)

        #new hidden state
        h = os * np.tanh(c)

        #predicted word for timestamp t
        yt = Why.dot(h)+by
        # forward pass again, but we do not have to store the activations now
        
        #softmax
        p = np.exp(yt) / np.sum(np.exp(yt))
        ix = np.random.choice(range(vocab_size), p=p.ravel())
        index = ix
        x = np.zeros((vocab_size, 1))
        x[index] = 1
        ixes.append(index)

    return ixes


if option == 'train':

    n, p = 0, 0
    n_updates = 0

    # momentum variables for Adagrad
    mWex, mWhy = np.zeros_like(Wex), np.zeros_like(Why)
    mby = np.zeros_like(by)

    mWf, mWi, mWo, mWc = np.zeros_like(Wf), np.zeros_like(Wi), np.zeros_like(Wo), np.zeros_like(Wc)
    mbf, mbi, mbo, mbc = np.zeros_like(bf), np.zeros_like(bi), np.zeros_like(bo), np.zeros_like(bc)

    smooth_loss = -np.log(1.0 / vocab_size) * seq_length  # loss at iteration 0

    data_length = cut_stream.shape[1]

    while True:
        # prepare inputs (we're sweeping from left to right in steps seq_length long)
        if p + seq_length + 1 >= data_length or n == 0:
            hprev = np.zeros((hidden_size, batch_size))  # reset RNN memory
            cprev = np.zeros((hidden_size, batch_size))
            p = 0  # go from start of data

        inputs = cut_stream[:, p:p + seq_length].T
        targets = cut_stream[:, p + 1:p + 1 + seq_length].T

        # sample from the model now and then
        if n % 1000 == 0:
            h_zero = np.zeros((hidden_size, 1))  # reset RNN memory
            c_zero = np.zeros((hidden_size, 1))
            sample_ix = sample((h_zero, c_zero), inputs[0][0], 1500)
            txt = ''.join(ix_to_char[ix] for ix in sample_ix)
            print('----\n %s \n----' % (txt,))

        # forward seq_length characters through the net and fetch gradient
        loss, activations, memory = forward(inputs, targets, (hprev, cprev))
        hprev, cprev = memory
        gradients = backward(activations)

        dWex, dWf, dWi, dWo, dWc, dbf, dbi, dbo, dbc, dWhy, dby = gradients
        smooth_loss = smooth_loss * 0.999 + loss/batch_size * 0.001
        if n % 10 == 0:
            print('iter %d, loss: %f' % (n, smooth_loss))  # print progress

        # perform parameter update with Adagrad
        for param, dparam, mem in zip([Wf, Wi, Wo, Wc, bf, bi, bo, bc, Wex, Why, by],
                                      [dWf, dWi, dWo, dWc, dbf, dbi, dbo, dbc, dWex, dWhy, dby],
                                      [mWf, mWi, mWo, mWc, mbf, mbi, mbo, mbc, mWex, mWhy, mby]):
            mem += dparam * dparam
            param += -learning_rate * dparam / np.sqrt(mem + 1e-8)  # adagrad update

        p += seq_length  # move data pointer
        n += 1  # iteration counter
        n_updates += 1
        if n_updates >= max_updates:
            break



elif option == "test":
    n, p = 0, 0
    n_updates = 0

    # momentum variables for Adagrad
    mWex, mWhy = np.zeros_like(Wex), np.zeros_like(Why)
    mby = np.zeros_like(by)

    mWf, mWi, mWo, mWc = np.zeros_like(Wf), np.zeros_like(Wi), np.zeros_like(Wo), np.zeros_like(Wc)
    mbf, mbi, mbo, mbc = np.zeros_like(bf), np.zeros_like(bi), np.zeros_like(bo), np.zeros_like(bc)

    smooth_loss = -np.log(1.0 / vocab_size) * seq_length  # loss at iteration 0

    data_length = cut_stream.shape[1]

    while True:
        # prepare inputs (we're sweeping from left to right in steps seq_length long)
        if p + seq_length + 1 >= data_length or n == 0:
            hprev = np.zeros((hidden_size, batch_size))  # reset RNN memory
            cprev = np.zeros((hidden_size, batch_size))
            p = 0  # go from start of data

        inputs = cut_stream[:, p:p + seq_length].T
        targets = cut_stream[:, p + 1:p + 1 + seq_length].T

        # forward seq_length characters through the net and fetch gradient
        loss, activations, memory = forward(inputs, targets, (hprev, cprev))
        hprev, cprev = memory
        backward(activations=activations)
        n_updates += 1
        if n_updates >= max_updates:
            break


elif option == 'gradcheck':

    data_length = cut_stream.shape[1]

    p = 0
    # inputs = [char_to_ix[ch] for ch in data[p:p + seq_length]]
    # targets = [char_to_ix[ch] for ch in data[p + 1:p + seq_length + 1]]
    inputs = cut_stream[:, p:p + seq_length].T
    targets = cut_stream[:, p + 1:p + 1 + seq_length].T

    delta = 0.0001

    hprev = np.zeros((hidden_size, batch_size))
    cprev = np.zeros((hidden_size, batch_size))

    memory = (hprev, cprev)

    loss, activations, hprev = forward(inputs, targets, memory)
    gradients = backward(activations, clipping=False)
    dWex, dWf, dWi, dWo, dWc, dbf, dbi, dbo, dbc, dWhy, dby = gradients

    for weight, grad, name in zip([Wf, Wi, Wo, Wc, bf, bi, bo, bc, Wex, Why, by],
                                  [dWf, dWi, dWo, dWc, dbf, dbi, dbo, dbc, dWex, dWhy, dby],
                                  ['Wf', 'Wi', 'Wo', 'Wc', 'bf', 'bi', 'bo', 'bc', 'Wex', 'Why', 'by']):

        str_ = ("Dimensions dont match between weight and gradient %s and %s." % (weight.shape, grad.shape))
        assert (weight.shape == grad.shape), str_

        print(name)
        print(weight.size)
        countidx = 0
        gradnumsum = 0
        gradanasum = 0
        relerrorsum = 0
        erroridx = []

        for i in range(weight.size):
            

            # evaluate cost at [x + delta] and [x - delta]
            w = weight.flat[i]
            weight.flat[i] = w + delta
            loss_positive, _, _ = forward(inputs, targets, memory)
            weight.flat[i] = w - delta
            loss_negative, _, _ = forward(inputs, targets, memory)
            weight.flat[i] = w  # reset old value for this parameter
            # fetch both numerical and analytic gradient
            grad_analytic = grad.flat[i]
            grad_numerical = (loss_positive - loss_negative) / (2 * delta)
            gradnumsum += grad_numerical
            gradanasum += grad_analytic
            rel_error = abs(grad_analytic - grad_numerical) / abs(grad_numerical + grad_analytic)
            if rel_error is None:
                rel_error = 0.
            relerrorsum += rel_error

            if rel_error > 0.001:
                #print ('WARNING %f, %f => %e ' % (grad_numerical, grad_analytic, rel_error))
                countidx += 1
                erroridx.append(i)
                
        print('For %s found %i bad gradients; with %i total parameters in the vector/matrix!' % (
            name, countidx, weight.size))
        print(' Average numerical grad: %0.9f \n Average analytical grad: %0.9f \n Average relative grad: %0.9f' % (
            gradnumsum / float(weight.size), gradanasum / float(weight.size), relerrorsum / float(weight.size)))
        print(' Indizes at which analytical gradient does not match numerical:', erroridx)
