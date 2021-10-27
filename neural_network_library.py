#!/usr/bin/python
# -*- coding: UTF-8 -*-

from random import random
from math import exp, log2
import matplotlib.pyplot as plt
import numpy as np

def _initialize_network(n_inputs, n_hidden, hidden_af, n_outputs, output_af, beta1, beta2, debug):
    assert n_hidden >= n_outputs, 'n_hidden < n_outputs: may result in information loss.'
    adam = beta1 is not None and beta2 is not None # Adam or SGD
    network = list()
    hidden_layer = [{'weights': [random() for i in range(n_inputs)]} for i in range(n_hidden)]
    for neuron in hidden_layer:
        neuron['bias'] = random()
        neuron['activation_fct'] = hidden_af
        if adam:
            neuron['mu'] = 0. # Gradient mean.
            neuron['nu'] = 0. # Gradient variance.
    network.append(hidden_layer)
    output_layer = [{'weights': [random() for i in range(n_hidden)]} for i in range(n_outputs)]
    for output_neuron in output_layer:
        output_neuron['bias'] = random()
        output_neuron['activation_fct'] = output_af
        if adam:
            output_neuron['mu'] = 0. # Gradient mean.
            output_neuron['nu'] = 0. # Gradient variance.
    network.append(output_layer)
    _initialize_network_debug(network, debug)
    _print_network(network)
    return network

def _print_network(network):
    print('network initialised')
    for idxl, layer in enumerate(network):
        print('    layer %s' % idxl)
        for idxn, neuron in enumerate(layer):
            print('        neuron %s:' % idxn)
            for key in neuron:
                print('            %s:' % key, neuron[key])

def _initialize_network_debug(network, debug):
    if debug:
        for layer in network:
            for neuron in layer:
                neuron['debug_delta'] = []
                neuron['debug_error'] = []
                neuron['debug_gradient'] = []
                neuron['debug_weights'] = []
                neuron['debug_bias'] = []
        for output_neuron in output_layer:
            output_neuron['debug_loss'] = []

def _neuron_activate(weights, bias, inputs):
    activation = bias # Bias (with associated input = 1.).
    for i in range(len(weights)):
        activation += weights[i] * inputs[i]
    return activation

def _transfer_sigmoid(activation):
    return 1.0 / (1.0 + exp(-activation))

def _transfer_derivative_sigmoid(output):
    return output * (1.0 - output)

def _transfer_relu(activation):
    if activation > 0.:
        return activation
    return 0.

def _transfer_derivative_relu(activation):
    if activation > 0.:
        return 1.
    return 0.

def _transfer(activation, activation_fct):
    if activation_fct == 'sigmoid':
        return _transfer_sigmoid(activation)
    elif activation_fct == 'relu':
        return _transfer_relu(activation)
    else:
        assert True, 'Error: unknown transfer fonction.'

def _transfer_derivative(activation, activation_fct):
    if activation_fct == 'sigmoid':
        return _transfer_derivative_sigmoid(activation)
    elif activation_fct == 'relu':
        return _transfer_derivative_relu(activation)
    else:
        assert True, 'Error: unknown transfer fonction.'

def _forward_propagate(network, data):
    inputs = data
    for layer in network:
        new_inputs = []
        for neuron in layer:
            activation = _neuron_activate(neuron['weights'], neuron['bias'], inputs)
            neuron['output'] = _transfer(activation, neuron['activation_fct'])
            new_inputs.append(neuron['output'])
        inputs = new_inputs
    return inputs

def _cross_entropy(expected, predicted, eps=1e-15):
    return -sum([expected[i]*log2(predicted[i]+eps) for i in range(len(expected))])

def _softmax(vector):
    e = np.exp(vector)
    return e / e.sum()

def _compute_loss(output_layer, expected, dbg):
    for j in range(len(output_layer)):
        output_neuron = output_layer[j]
        error = output_neuron['output'] - expected[j] # error = output - expected <=> GD with minus (-= alpha * grad).
        output_neuron['loss'] = error
        if dbg:
            output_neuron['debug_loss'].append(output_neuron['loss'])

    output_loss = [output_neuron['loss'] for output_neuron in output_layer]
    softmax_loss = _softmax(output_loss) # Need softmax (all outputs > 0) before cross entropy (log).
    loss = _cross_entropy(expected, softmax_loss)
    return loss

def _adam(neuron, beta1, beta2, time):
    adam = beta1 is not None and beta2 is not None # Adam or SGD
    if not adam:
        return # Gradient does not need to be modified.
    grad, mu, nu = neuron['gradient'], neuron['mu'], neuron['nu']
    mu = beta1 * mu + (1. - beta1) * grad    # Gradient mean.
    nu = beta2 * nu + (1. - beta2) * grad**2 # Gradient variance.
    neuron['mu'] = mu # Modify gradient mean.
    neuron['nu'] = nu # Modify gradient variance.
    mu_hat = mu / (1. - beta1**(time+1)) # Gradient mean correction.
    nu_hat = nu / (1. - beta2**(time+1)) # Gradient variance correction.
    time += 1 # Update time for Adam gradient descent.
    eps = 1e-8 # Make sure we never divide by zero.
    neuron['gradient'] = mu_hat / (np.sqrt(nu_hat) + eps) # Modify gradient.

def _backward_propagate_error(network, beta1, beta2, time, debug):
    for i in reversed(range(len(network))): # Looping backward from output to hidden layer.
        layer = network[i]
        if i == len(network)-1: # Output layer.
            for neuron in layer:
                neuron['error'] = neuron['loss'] # Initialize error with loss.
                neuron['gradient'] = 1. # Score s such that ds/ds = 1.
        else: # Hidden layer.
            for j in range(len(layer)):
                error, next_layer = 0., network[i + 1]
                for next_neuron in next_layer: # Looping over hidden layer output.
                    error += (next_neuron['weights'][j] * next_neuron['delta'])
                neuron = layer[j]
                neuron['error'] = error
                neuron['gradient'] = _transfer_derivative(neuron['output'], neuron['activation_fct'])
        for j in range(len(layer)):
            neuron = layer[j]
            _adam(neuron, beta1, beta2, time)
            neuron['delta'] = neuron['error'] * neuron['gradient']
            if debug:
                neuron['debug_delta'].append(neuron['delta'])
                neuron['debug_error'].append(neuron['error'])
                neuron['debug_gradient'].append(neuron['gradient'])

def _update_weights(network, data, alpha, debug):
    for i in range(len(network)):
        inputs = data
        if i != 0:
            prev_layer = network[i - 1] # Looping over hidden layer input.
            inputs = [neuron['output'] for neuron in prev_layer]
        for neuron in network[i]:
            for j in range(len(inputs)):
                neuron['weights'][j] -= alpha * neuron['delta'] * inputs[j]
            neuron['bias'] -= alpha * neuron['delta'] # Bias (with associated input = 1.).
            if debug:
                neuron['debug_weights'].append(neuron['weights'])
                neuron['debug_bias'].append(neuron['bias'])

def _network_debug_plot(network):
    for data in ['debug_error', 'debug_delta', 'debug_gradient']:
        _, axes = plt.subplots(max([len(layer) for layer in network]), len(network))
        plt.suptitle('neural network %s' % data)
        for idxl, layer in enumerate(network):
            for idxn, neuron in enumerate(layer):
                axes[idxn][idxl].set_title('layer %s, neuron %s' % (idxl, idxn))
                axes[idxn][idxl].plot(neuron[data], label=data)
                axes[idxn][idxl].legend()
    _, axes = plt.subplots(max([len(layer) for layer in network]), len(network))
    plt.suptitle('neural network debug_weights/debug_bias')
    for idxl, layer in enumerate(network):
        for idxn, neuron in enumerate(layer):
            axes[idxn][idxl].set_title('layer %s, neuron %s' % (idxl, idxn))
            for idxw in range(len(neuron['weights'])):
                debug_weights = [weights[idxw] for weights in neuron['debug_weights']]
                axes[idxn][idxl].plot(debug_weights, label=('weights %s' % idxw))
            axes[idxn][idxl].plot(neuron['debug_bias'], label='bias')
            axes[idxn][idxl].legend()
    output_layer = network[-1]
    _, axes = plt.subplots(len(output_layer), 1)
    plt.suptitle('neural network debug_loss')
    for idxn, output_neuron in enumerate(output_layer):
        axes[idxn].set_title('output layer, neuron %s' % idxn)
        axes[idxn].plot(output_neuron['debug_loss'], label='loss')
        axes[idxn].legend()

def _network_metrics(network, metrics, train_set, val_set):
    _, train_error = network_evaluate(train_set, network)
    metrics['train_error'].append(train_error)
    _, val_error = network_evaluate(val_set, network)
    metrics['val_error'].append(val_error)
    return train_error, val_error

def make_batch(iterable, batch_size=16):
    n = len(iterable)
    for i in range(0, n, batch_size):
        yield iterable[i:min(i + batch_size, n)]

def network_train(train_set, val_set,
                  n_hidden, hidden_af, n_outputs, output_af, n_epoch,
                  alpha, beta1, beta2, # If beta1 = beta2 = None, we get SGD instead of Adam.
                  batch_size=16, debug=False):
    n_inputs = len(train_set[0]) - 1 # All data but not the target (associated to the data).
    network = _initialize_network(n_inputs, n_hidden, hidden_af, n_outputs, output_af, beta1, beta2, debug)
    print('network training')
    metrics = {'train_error': [], 'val_error': []}
    time = 1
    for epoch in range(n_epoch):
        loss = 0.
        batches = list(make_batch(train_set, batch_size=batch_size))
        for idxb, batch in enumerate(batches):
            for idxr, row in enumerate(batch): # First backpropagate
                data, target = row[:-1], row[-1]
                outputs = _forward_propagate(network, data)
                expected = [0 for i in range(n_outputs)]
                expected[target] = 1
                dbg = True if debug and idxb == len(batches) - 1 and idxr == len(batch) - 1 else False
                loss += _compute_loss(network[-1], expected, dbg)
                _backward_propagate_error(network, beta1, beta2, time, dbg)
            for idxr, row in enumerate(batch): # Then update model: update neurons weights.
                data = row[:-1]
                dbg = True if debug and idxb == len(batches) - 1 and idxr == len(batch) - 1 else False
                _update_weights(network, data, alpha, dbg)
        train_error, val_error = _network_metrics(network, metrics, train_set, val_set)
        print('    epoch=%03d, alpha=%.3f, loss=%.3f, train_error=%.3f%%, val_error=%.3f%%' % (epoch, alpha, loss, train_error, val_error))
    if debug:
        _network_debug_plot(network)
    return network, metrics

def network_predict(network, row):
    data = row[:-1]
    outputs = _forward_propagate(network, data)
    return outputs.index(max(outputs))

def network_evaluate(data_set, network):
    predictions, good_predictions = [], 0
    for row in data_set:
        target = row[-1]
        predicted_row = network_predict(network, row)
        if predicted_row == target:
            good_predictions += 1
        predictions.append(predicted_row)
    error = (len(data_set) - good_predictions)*100./len(data_set) # Error %
    return predictions, error
