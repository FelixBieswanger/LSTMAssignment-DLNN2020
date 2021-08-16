"""
Minimal character-level LSTM model. Written by Ngoc Quan Pham
Code structure borrowed from the Vanilla RNN model from Andreij Karparthy @karparthy.
BSD License
"""
import numpy as np
from random import uniform
import sys
from icecream import ic

class dataset:

    def __init__(self, data_path):

        # data I/O
        # should be simple plain text file. The sample from "Hamlet - Shakespeares" is provided in data/
        with open(data_path, 'r') as f:
            data = f.read()
        chars = sorted(list(set(data)))  # added sorted so that the character list is deterministic
        #print(chars)
        self.data_size, self.vocab_size = len(data), len(chars)
        #print('data has %d characters, %d unique.' % (self.data_size, self.vocab_size))
        self.char_to_ix = {ch: i for i, ch in enumerate(chars)}
        self.ix_to_char = {i: ch for i, ch in enumerate(chars)}

        # this will load the data into memory
        self.data_stream = np.asarray([self.char_to_ix[char] for char in data])
        #print(self.data_stream.shape)


    def get_cut_stream(self,seq_length, batch_size):
        bound = (self.data_stream.shape[0] // (seq_length * batch_size)) * (seq_length * batch_size)
        cut_stream = self.data_stream[:bound]
        cut_stream = np.reshape(cut_stream, (batch_size, -1))

        return cut_stream

    def get_ix_to_char(self):
        return self.ix_to_char

    def get_char_to_ix(self):
        return self.char_to_ix

    def get_data_and_vocab_size(self):
        return self.data_size, self.vocab_size

class lstm():

    def __init__(self, emb_size,hidden_size,seq_length,learning_rate,max_updates,batch_size,std,vocab_size):
        # hyperparameters
        self.emb_size = emb_size
        self.hidden_size = hidden_size  # size of hidden layer of neurons
        self.seq_length = seq_length  # number of steps to unroll the RNN for
        self.learning_rate = learning_rate
        self.max_updates = max_updates
        self.batch_size = batch_size
        self.std = std
        self.vocab_size = vocab_size

        self.concat_size = self.emb_size + self.hidden_size


        
        # model parameters
        # char embedding parameters
        self.Wex = np.random.randn(self.emb_size, vocab_size) * self.std  # embedding layer

        # LSTM parameters
        self.Wf = np.random.randn(self.hidden_size, self.concat_size) * self.std  # forget gate
        self.Wi = np.random.randn(self.hidden_size, self.concat_size) * self.std  # input gate
        self.Wo = np.random.randn(self.hidden_size, self.concat_size) * self.std  # output gate
        self.Wc = np.random.randn(self.hidden_size, self.concat_size) * self.std  # c term

        self.bf = np.zeros((self.hidden_size, 1))  # forget bias
        self.bi = np.zeros((self.hidden_size, 1))  # input bias
        self.bo = np.zeros((self.hidden_size, 1))  # output bias
        self.bc = np.zeros((self.hidden_size, 1))  # memory bias

        # Output layer parameters
        self.Why = np.random.randn(vocab_size, self.hidden_size) * self.std  # hidden to output
        self.by = np.random.randn(vocab_size, 1) * self.std  # output bias


    def sigmoid(self,x):
        return 1 / (1 + np.exp(-x))


    def dsigmoid(self,y):
        return y * (1 - y)


    def dtanh(self,x):
        return 1 - x * x


    # The numerically stable softmax implementation
    def softmax(self,x):
        # assuming x shape is [feature_size, batch_size]
        e_x = np.exp(x - np.max(x, axis=0))
        return e_x / e_x.sum(axis=0)

    def forget_gate(self, zs):
        #f = sigmoid(Wf * z + bf)
        return self.sigmoid(np.dot(self.Wf,zs) + self.bf)

    def input_gate(self, zs):
        # i = sigmoid(Wi * z + bi)
        return self.sigmoid(np.dot(self.Wi,zs) + self.bi)

    def candidate_content(self, zs):
        #c_ = tanh(Wc * z + bc)
        return np.tanh(np.dot(self.Wc,zs) + self.bc)

    def compute_cell_content(self, fs,c_t,ins, cc):
        #ic("compute cell content", fs.shape, c_t.shape, ins.shape, cc.shape)
        return fs * c_t + ins * cc

    def output_gate(self,z):
        #o = sigmoid(Wo * z + bo)
        return self.sigmoid(np.dot(self.Wo,z) + self.bo)

    def compute_cell_state(self, o, c_t):
        #ic("compute cell state", o.shape, c_t.shape)
        return o * np.tanh(c_t)

    def forward(self,func_inputs, targets, memory):
        """
        inputs,targets are both list of integers.
        hprev is Hx1 array of initial hidden state
        returns the loss, gradients on model parameters, and last hidden state
        """
        # xs: inputs
        # wes: Word embeddings at timestamp
        # zs: concatenated input and h
        # fs: forget_states
        # ins: input gate state at timestamp
        # cc: candidate content
        # c_t: cell content 
        # o: output gate
        # hs: cell state
        # outputs
        # ps: softmax output
        # ls: label as one hot vector

        hprev, cprev = memory
        #xs, wes, zs,fs, ins, cc, c_t, o, hs, ps, ls = {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}
        inputs, embeddings, input_and_h_concatenateds, forget_states, input_gate_states = {}, {}, {}, {}
        candidate_contents, cell_contents, output_gate_values, h_cell_state, outputs = {}, {}, {}, {}, {}, {}
        softmax_outputs, labels = {}, {}
        #hs, ys, ps, cs, zs,  c_s, ls =  , {}

        #hs[-1] = np.copy(hprev)
        #c_t[-1] = np.copy(cprev)
        h_cell_state[-1] = np.copy(hprev)
        cell_contents[-1] = np.copy(cprev)



        loss = 0
        input_length = func_inputs.shape[0]

        # forward pass
        for t in range(input_length):
            #xs[t] = np.zeros((self.vocab_size, self.batch_size))  
            inputs[t] = np.zeros((self.vocab_size, self.batch_size))# encode in 1-of-k representation
            for b in range(self.batch_size):
                inputs[t][func_inputs[t][b]][b] = 1
            
            #ic(xs[t].shape)
            # convert word indices to word embeddings
            #wes[t] = np.dot(self.Wex, xs[t])
            embeddings[t] = np.dot(self.Wex, inputs[t])
            #ic(wes[t].shape)
            #ic(hs[t-1].shape)
            # LSTM cell operation
            # first concatenate the input and h to get z
            #zs[t] = np.row_stack((hs[t - 1], wes[t]))
            input_and_h_concatenateds[t] = np.row_stack((h_cell_state[t - 1], embeddings[t]))
            #ic(zs[t].shape)
            # compute the forget gate
            # f = sigmoid(Wf * z + bf)
            #fs[t] = self.forget_gate(zs[t])
            forget_states[t] = self.forget_gate(input_and_h_concatenateds[t])
            #ic(fs[t].shape)
            # compute the input gate
            # i = sigmoid(Wi * z + bi)
            #ins[t] = self.input_gate(zs[t])
            input_gate_states[t] = self.input_gate(input_and_h_concatenateds[t])
            #ic(ins[t].shape)
            # compute the candidate memory
            #c_ = tanh(Wc * z + bc)
            #cc[t] = self.candidate_content(zs[t])
            candidate_contents[t] = self.candidate_content(input_and_h_concatenateds[t])
            #ic(cc[t].shape)
            # new memory: applying forget gate on the previous memory
            # and then adding the input gate on the candidate memory
            # c_t = f * c_(t-1) + i * c_
            #c_t[t] = self.compute_cell_content(fs[t],c_t[t-1],ins[t], cc[t])
            cell_contents[t] = self.compute_cell_content(forget_states[t],cell_contents[t-1],input_gate_states[t], candidate_contents[t])
            #ic(c_t[t].shape)
            # output gate
            #o = sigmoid(Wo * z + bo)
            #o[t] = self.output_gate(zs[t])
            output_gate_values[t] = self.output_gate(input_and_h_concatenateds[t])
            #ic(o[t].shape)
            #cell state
            #hs[t] = self.compute_cell_state(o[t], c_t[t])
            h_cell_state = self.compute_cell_state(output_gate_values[t], cell_contents[t])
            #ic(hs[t].shape)
            # DONE LSTM
            # output layer - softmax and cross-entropy loss
            #output = np.dot(self.Why,hs[t]) + self.by
            outputs[t] = np.dot(self.Why,h_cell_state[t]) + self.by

            # unnormalized log probabilities for next chars
            # softmax for probabilities for next chars
            #ps[t] = self.softmax(output)
            softmax_outputs[t] = self.softmax(outputs[t])
            #ic(ps[t].shape)
            # label (also one hot vector)
            #ls[t] = np.zeros((self.vocab_size, self.batch_size))
            labels[t] = np.zeros((self.vocab_size, self.batch_size))
            for b in range(self.batch_size):
                #ls[t][targets[t][b]][b] = 1
                labels[t][targets[t][b]][b] = 1
            #ic(ls[t].shape)
            # cross-entropy loss
            #ic("loss Calculation", ps[t].shape, ls[t].shape)
            #loss_t = np.sum(- np.log(ps[t])  * ls[t])
            loss_t = np.sum(- np.log(softmax_outputs[t])  * labels[t])
            loss += loss_t
            # loss += -np.log(ps[t][targets[t],0])

        #activations = (xs, wes, zs,fs, ins, cc, c_t, o, hs, ps, ls)
        activations = (inputs, embeddings, input_and_h_concatenateds, forget_states, input_gate_states, candidate_contents, cell_contents, output_gate_values, h_cell_state, outputs, softmax_outputs, labels)

        #memory = (hs[input_length - 1], c_t[input_length -1])
        memory = (h_cell_state[input_length - 1], cell_contents[input_length -1])
        return loss, activations, memory

    def backward(self,activations, clipping=True):
        xs, wes, hs, ys, ps, cs, zs, ins, c_s, ls, os, fs = activations
        inputs, embeddings, input_and_h_concatenateds, forget_states, input_gate_states, candidate_contents, cell_contents, output_gate_values, h_cell_state, outputs, softmax_outputs, labels = activations

        # backward pass: compute gradients going backwards
        # Here we allocate memory for the gradients
        dWex, dWhy = np.zeros_like(self.Wex), np.zeros_like(self.Why)
        dby = np.zeros_like(self.by)
        dWf, dWi, dWc, dWo = np.zeros_like(self.Wf), np.zeros_like(self.Wi), np.zeros_like(self.Wc), np.zeros_like(self.Wo)
        dbf, dbi, dbc, dbo = np.zeros_like(self.bf), np.zeros_like(self.bi), np.zeros_like(self.bc), np.zeros_like(self.bo)

        dhnext = np.zeros_like(hs[0])
        dcnext = np.zeros_like(cs[0])

        input_length = len(xs)

        # back propagation through time starts here
        for t in reversed(range(input_length)):
            # computing the gradients here
            pass
        # clip to mitigate exploding gradients
        if clipping:
            for dparam in [dWex, dWf, dWi, dWo, dWc, dbf, dbi, dbo, dbc, dWhy, dby]:
                np.clip(dparam, -5, 5, out=dparam)

        gradients = (dWex, dWf, dWi, dWo, dWc, dbf, dbi, dbo, dbc, dWhy, dby)

        return gradients


    def sample(self,memory, seed_ix, n):
        """
        sample a sequence of integers from the model
        h is memory state, seed_ix is seed letter for first time step
        """
        h, c = memory
        x = np.zeros((self.vocab_size, 1))
        x[seed_ix] = 1
        ixes = []
        for t in range(n):

            # forward pass again, but we do not have to store the activations now
            p = self.softmax(h)
            #p = np.exp(y) / np.sum(np.exp(y))
            ic(vocab_size)
            ic(p.shape)
            ix = np.random.choice(range(self.vocab_size), p=p.ravel())

            index = ix
            x = np.zeros((self.vocab_size, 1))
            x[index] = 1
            ixes.append(index)
        return ixes





# hyperparameters
emb_size = 16
hidden_size = 256  # size of hidden layer of neurons
seq_length = 128  # number of steps to unroll the RNN for
learning_rate = 5e-2
#max_updates = 500000
max_updates = 100
batch_size = 32
std = 0.1



option = sys.argv[1]
data = dataset("data/input.txt")
data_size, vocab_size = data.get_data_and_vocab_size()
char_to_ix = data.get_char_to_ix()
ix_to_char = data.get_ix_to_char()
cut_stream = data.get_cut_stream(seq_length, batch_size)

model = lstm(emb_size,hidden_size,seq_length,learning_rate,max_updates,batch_size,std,vocab_size)


if option == 'test':
    n, p = 0, 0
    n_updates = 0

    # momentum variables for Adagrad
    mWex, mWhy = np.zeros_like(model.Wex), np.zeros_like(model.Why)
    mby = np.zeros_like(model.by)

    mWf, mWi, mWo, mWc = np.zeros_like(model.Wf), np.zeros_like(model.Wi), np.zeros_like(model.Wo), np.zeros_like(model.Wc)
    mbf, mbi, mbo, mbc = np.zeros_like(model.bf), np.zeros_like(model.bi), np.zeros_like(model.bo), np.zeros_like(model.bc)

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
        if n % 200 == 1:
            h_zero = np.zeros((hidden_size, 1))  # reset RNN memory
            c_zero = np.zeros((hidden_size, 1))
            sample_ix = model.sample((h_zero, c_zero), inputs[0][0], 2000)
            txt = ''.join(ix_to_char[ix] for ix in sample_ix)
            print('----\n %s \n----' % (txt,))

        # forward seq_length characters through the net and fetch gradient
        loss, activations, memory = model.forward(inputs, targets, (hprev, cprev))
        #ic(len(activations[7]))
        hprev, cprev = memory
        n_updates += 1
        if n_updates >= max_updates:
            break

if option == 'train':

    n, p = 0, 0
    n_updates = 0

    # momentum variables for Adagrad
    mWex, mWhy = np.zeros_like(model.Wex), np.zeros_like(model.Why)
    mby = np.zeros_like(model.by)

    mWf, mWi, mWo, mWc = np.zeros_like(model.Wf), np.zeros_like(model.Wi), np.zeros_like(model.Wo), np.zeros_like(model.Wc)
    mbf, mbi, mbo, mbc = np.zeros_like(model.bf), np.zeros_like(model.bi), np.zeros_like(model.bo), np.zeros_like(model.bc)

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
        if n % 200 == 0:
            h_zero = np.zeros((hidden_size, 1))  # reset RNN memory
            c_zero = np.zeros((hidden_size, 1))
            sample_ix = model.sample((h_zero, c_zero), inputs[0][0], 2000)
            txt = ''.join(ix_to_char[ix] for ix in sample_ix)
            print('----\n %s \n----' % (txt,))

        # forward seq_length characters through the net and fetch gradient
        loss, activations, memory = model.forward(inputs, targets, (hprev, cprev))
        hprev, cprev = memory
        gradients = model.backward(activations)

        dWex, dWf, dWi, dWo, dWc, dbf, dbi, dbo, dbc, dWhy, dby = gradients
        smooth_loss = smooth_loss * 0.999 + loss/batch_size * 0.001
        if n % 20 == 0:
            print('iter %d, loss: %f' % (n, smooth_loss))  # print progress

        # perform parameter update with Adagrad
        for param, dparam, mem in zip([model.Wf, model.Wi, model.Wo, model.Wc, model.bf, model.bi, model.bo, model.bc, model.Wex, model.Why, model.by],
                                      [dWf, dWi, dWo, dWc, dbf, dbi, dbo, dbc, dWex, dWhy, dby],
                                      [mWf, mWi, mWo, mWc, mbf, mbi, mbo, mbc, mWex, mWhy, mby]):
            mem += dparam * dparam
            param += -learning_rate * dparam / np.sqrt(mem + 1e-8)  # adagrad update

        p += seq_length  # move data pointer
        n += 1  # iteration counter
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
                print ('WARNING %f, %f => %e ' % (grad_numerical, grad_analytic, rel_error))
                countidx += 1
                erroridx.append(i)
                
        print('For %s found %i bad gradients; with %i total parameters in the vector/matrix!' % (
            name, countidx, weight.size))
        print(' Average numerical grad: %0.9f \n Average analytical grad: %0.9f \n Average relative grad: %0.9f' % (
            gradnumsum / float(weight.size), gradanasum / float(weight.size), relerrorsum / float(weight.size)))
        print(' Indizes at which analytical gradient does not match numerical:', erroridx)
