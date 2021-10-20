#!/usr/bin/python
# -*- coding: UTF-8 -*-

from random import random
from math import exp, log2
import matplotlib.pyplot as plt
import numpy as np

def _initialize_network(n_inputs, n_hidden, hidden_af, n_outputs, output_af, debug):
    assert n_hidden >= n_outputs, 'n_hidden < n_outputs: may result in information loss.'
    network = list()
    hidden_layer = [{'weights':[random() for i in range(n_inputs)]} for i in range(n_hidden)]
    for neuron in hidden_layer:
        neuron['bias'] = random()
        neuron['activation_fct'] = hidden_af
    network.append(hidden_layer)
    output_layer = [{'weights':[random() for i in range(n_hidden)]} for i in range(n_outputs)]
    for neuron in output_layer:
        neuron['bias'] = random()
        neuron['activation_fct'] = output_af
    network.append(output_layer)
    print('network initialised')
    for idxl, layer in enumerate(network):
        print('    layer %s' % idxl)
        for idxn, neuron in enumerate(layer):
            print('        neuron %s' % idxn, neuron)
    if debug:
        for layer in network:
            for neuron in layer:
                neuron['debug_delta'] = []
                neuron['debug_error'] = []
                neuron['debug_gradient'] = []
                neuron['debug_weights'] = []
                neuron['debug_bias'] = []
    return network

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

def _compute_loss(output_layer, expected):
    for j in range(len(output_layer)):
        output_neuron = output_layer[j]
        error = expected[j] - output_neuron['output']
        output_neuron['loss'] = error

    output_loss = [output_neuron['loss'] for output_neuron in output_layer]
    softmax_loss= _softmax(output_loss)
    loss = _cross_entropy(expected, softmax_loss)
    return loss

def _backward_propagate_error(network, expected, debug):
    for i in reversed(range(len(network))): # Looping backward from output to hidden layer.
        layer = network[i]
        if i == len(network)-1: # Output layer.
            for neuron in layer:
                neuron['error'] = neuron['loss'] # Initialize error with loss.
        else: # Hidden layer.
            for j in range(len(layer)):
                error, next_layer = 0., network[i + 1]
                for neuron in next_layer: # Looping over hidden layer output.
                    error += (neuron['weights'][j] * neuron['delta'])
                layer[j]['error'] = error
        for j in range(len(layer)):
            neuron = layer[j]
            gradient = _transfer_derivative(neuron['output'], neuron['activation_fct'])
            neuron['delta'] = neuron['error'] * gradient
            if debug:
                neuron['debug_delta'].append(neuron['delta'])
                neuron['debug_error'].append(neuron['error'])
                neuron['debug_gradient'].append(gradient)

def _update_weights(network, data, l_rate, debug):
    for i in range(len(network)):
        inputs = data
        if i != 0:
            prev_layer = network[i - 1] # Looping over hidden layer input.
            inputs = [neuron['output'] for neuron in prev_layer]
        for neuron in network[i]:
            for j in range(len(inputs)):
                neuron['weights'][j] += l_rate * neuron['delta'] * inputs[j]
            neuron['bias'] += l_rate * neuron['delta'] # Bias (with associated input = 1.).
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
                  n_hidden, hidden_af, n_outputs, output_af, n_epoch, l_rate,
                  batch_size=16, debug=False):
    n_inputs = len(train_set[0]) - 1 # All data but not the target (associated to the data).
    network = _initialize_network(n_inputs, n_hidden, hidden_af, n_outputs, output_af, debug)
    print('network training')
    metrics = {'train_error': [], 'val_error': []}
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
                loss += _compute_loss(network[-1], expected)
                _backward_propagate_error(network, expected, dbg)
            for idxr, row in enumerate(batch): # Then update model: update neurons weights.
                data = row[:-1]
                dbg = True if debug and idxb == len(batches) - 1 and idxr == len(batch) - 1 else False
                _update_weights(network, data, l_rate, dbg)
        train_error, val_error = _network_metrics(network, metrics, train_set, val_set)
        print('    epoch=%d, lrate=%.3f, loss=%.3f train_error=%.3f%%, val_error=%.3f%%' % (epoch, l_rate, loss, train_error, val_error))
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
