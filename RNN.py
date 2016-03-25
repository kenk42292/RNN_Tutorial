import numpy as np
import operator
import datetime
import sys
import pickle
from utils import *



class RNN:

    def __init__(self, word_dim, hidden_dim=100, bptt_truncate=4):
        #Assign instance variables
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.bptt_truncate = bptt_truncate
        #Randomly initialize network parameters
        self.U = np.random.uniform(-np.sqrt(1./word_dim), np.sqrt(1./word_dim), (hidden_dim, word_dim))
        self.V = np.random.uniform(-np.sqrt(1./word_dim), np.sqrt(1./word_dim), (word_dim, hidden_dim))
        self.W = np.random.uniform(-np.sqrt(1./word_dim), np.sqrt(1./word_dim), (hidden_dim, hidden_dim))       
        

    def forward_propagate(self, seq):
        T = len(seq) #Total number of time steps in sequence
        hidden_states = np.zeros((T+1, self.hidden_dim))
        hidden_states[-1] = np.zeros(self.hidden_dim)
        outputs = np.zeros((T, self.word_dim))
        for t in np.arange(T):
            hidden_states[t] = np.tanh(self.U[:,seq[t]] + self.W.dot(hidden_states[t-1]))
            outputs[t] = softmax(self.V.dot(hidden_states[t]))
        return [outputs, hidden_states]


    def predict(self, seq):
        """
        Perform forward propagation and return index of the highest score
        """
        outputs, hidden_states = self.forward_propagate(seq)
        return np.argmax(outputs, axis=1)


    def calculate_loss(self, x, y):
        # Take average, given total loss
        N = np.sum([len(y_i) for y_i in y])
        return self.calculate_total_loss(x,y)/N



    def calculate_total_loss(self, x, y):
        """
        For every word in every sentence, accumulate total loss
        """
        L = 0
        for i in np.arange(len(y)):
            o, s = self.forward_propagate(x[i])
            # o is an (n by 8000) matrix, where n=len(x[i])
            # Now get an array of our predictions of the correct words for the sentence
            correct_word_predictions = o[np.arange(len(x[i])), y[i]]
            # Add to the loss based on how off we were
            L += -1*np.sum(np.log(correct_word_predictions))
        return L


    def train(self, use_existing_model, X_train, Y_train, learning_rate=0.005, nepoch=100, evaluate_loss_after=10):
        
        if use_existing_model:
            with open('model_params.pickle', 'rU') as f:
                self.U, self.V, self.W = pickle.load(f)
        else:
            # Keep track of losses
            losses = []
            num_examples_seen = 0
            for epoch in range(nepoch):
                if (epoch%evaluate_loss_after==0):
                    loss = self.calculate_loss(X_train, Y_train)
                    losses.append((num_examples_seen, loss))
                    time = datetime.datetime.today().strftime("%Y-%m-%d %H:%M:%S")
                    print "%s: Loss after num_examples_seen=%d epoch=%d: %f" % (time, num_examples_seen, epoch, loss)
                    # Adjust the learning rate if loss increases
                    if (len(losses) > 1 and losses[-1][1] > losses[-2][1]):
                        learning_rate = learning_rate * 0.5
                        print "Setting learning rate to %f" % learning_rate
                    sys.stdout.flush()
                # For each training example...
                for i in range(len(Y_train)):
                    self.sgd_step(X_train[i], Y_train[i], learning_rate)
                    num_examples_seen += 1
            with open('model_params.pickle', 'w') as f:
                pickle.dump([self.U, self.V, self.W], f)

    
    def sgd_step(self, x, y, learning_rate):
        dLdU, dLdV, dLdW = self.bptt(x, y)
        self.U -= learning_rate * dLdU
        self.V -= learning_rate * dLdV
        self.W -= learning_rate * dLdW
                



    def bptt(self, x, y):
        T = len(y)
        #perform forward propagation
        o, s = self.forward_propagate(x)
        #We accumulate the gradients in these variables
        dLdU = np.zeros(self.U.shape)
        dLdV = np.zeros(self.V.shape)
        dLdW = np.zeros(self.W.shape)
        delta_o = o
        delta_o[np.arange(len(y)), y] -= 1
        #For each outpu backwards...
        for t in np.arange(T)[::-1]:
            dLdV += np.outer(delta_o[t], s[t].T)
            # Initial delta calculation
            delta_t = self.V.T.dot(delta_o[t]) * (1-s[t]**2)
            # Backpropagation through time (for at most self.bbtt_truncate steps)
            for bptt_step in np.arange(max(0, t-self.bptt_truncate), t+1)[::-1]:
                # print "Backpropagation step t=%d bptt step=%d " % (t, bptt_step)
                dLdW += np.outer(delta_t, s[bptt_step-1])
                dLdU[:,x[bptt_step]] += delta_t
                # Update delta for next step
                delta_t = self.W.T.dot(delta_t) * (1-s[bptt_step-1]**2)
        return [dLdU, dLdV, dLdW]


    def generate_sentence(self, unknown_token, sentence_start_token, sentence_end_token, index_to_word, word_to_index, min_sentence_length=5):
        sentence_length = 0
        while sentence_length < min_sentence_length:
            # Start the sentence with the start token
            new_sentence = [word_to_index[sentence_start_token]]
            # Repeat until we get an end token
            while not new_sentence[-1] == word_to_index[sentence_end_token]:
                next_word_probs = self.forward_propagate(new_sentence)[0]
                sampled_word = word_to_index[unknown_token]
                while sampled_word == word_to_index[unknown_token]:
                    samples = np.random.multinomial(1, next_word_probs[-1])
                    sampled_word = np.argmax(samples)
                new_sentence.append(sampled_word)
            sentence_str = [index_to_word[x] for x in new_sentence[1:-1]]
            sentence_length = len(sentence_str)
        return " ".join(sentence_str)




    def gradient_check(self, x, y, h=0.001, error_threshold=0.01):
        # Calculate the gradients using backpropagation. We want to check if these are correct
        bptt_gradients = self.bptt(x, y)
        # List of all parameters we want to check.
        model_parameters = ["U", "V", "W"]
        # Gradient check for each parameter
        for pidx, pname in enumerate(model_parameters):
            # Get the actual paramter value from the model. e.g. model.W
            parameter = operator.attrgetter(pname)(self)
            print "Performing gradient check for paramter %s with size %d." % (pname, np.prod(parameter.shape))
            # Iterate over each element of the parameter marix, e.g. (0, 0), (0, 1), ...
            iter = np.nditer(parameter, flags= ["multi_index"], op_flags=["readwrite"])
            while not iter.finished:
                index = iter.multi_index
                # Save the original value so we can reset it later
                original_value = parameter[index]
                # Estimate the gradient using (f(x+h) - f(x-h))/(2*h)
                parameter[index] = original_value+h
                gradplus = self.calculate_total_loss([x], [y])
                parameter[index] = original_value-h
                gradminus = self.calculate_total_loss([x], [y])
                estimated_gradient = (gradplus - gradminus) / (2*h)
                # Reset parameter to original value
                parameter[index] = original_value
                # The gradient for this parameter calculated using backpropagation
                backprop_gradient = bptt_gradients[pidx][index]
                # Calculate the relative error: (|x-y|/(|x|+|y|))
                relative_error = np.abs(backprop_gradient - estimated_gradient)/(np.abs(backprop_gradient) + np.abs(estimated_gradient))
                #If the eror is too large, fail the gradient check
                if relative_error > error_threshold:
                    print "Gradient Check ERROR: parameter=%s i=%s" % (pname, index)
                    print "+h Loss: %f" % gradplus
                    print "-h Loss: %f" % gradminus
                    print "Estimated_gadient: %f" % estimated_gradient
                    print "Backpropagation gradient: %f" % backprop_gradient
                    print "Relative Error: %f" % relative_error
                    return
                iter.iternext()
            print "Gradient check for parameter %s passed." % (pname)

                



