#!/usr/bin/python
# -*- coding: UTF-8 -*-

from random import random
from math import exp, log2
import matplotlib.pyplot as plt

def _initialize_network(n_inputs, n_hidden, n_outputs, debug):
    network = list()
    hidden_layer = [{'weights':[random() for i in range(n_inputs + 1)]} for i in range(n_hidden)] # +1 for bias.
    network.append(hidden_layer)
    output_layer = [{'weights':[random() for i in range(n_hidden + 1)]} for i in range(n_outputs)] # +1 for bias.
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
    return network

def _neuron_activate(weights, inputs):
    activation = weights[-1] # Bias (associated input = 1.) is the last weight in the list.
    for i in range(len(weights)-1): # Sum all but bias.
        activation += weights[i] * inputs[i]
    return activation

def _sigmoid_transfer(activation):
    return 1.0 / (1.0 + exp(-activation))

def _sigmoid_transfer_derivative(output):
    return output * (1.0 - output)

def _forward_propagate(network, row):
    inputs = row
    for layer in network:
        new_inputs = []
        for neuron in layer:
            activation = _neuron_activate(neuron['weights'], inputs)
            neuron['output'] = _sigmoid_transfer(activation)
            new_inputs.append(neuron['output'])
        inputs = new_inputs
    return inputs

def _cross_entropy(p, q, eps=1e-15):
    return -sum([p[i]*log2(q[i]+eps) for i in range(len(p))])

def _backward_propagate_error(network, expected, debug):
    for i in reversed(range(len(network))): # Looping backward from output to hidden layer.
        layer = network[i]
        errors = list()
        if i != len(network)-1: # Hidden layer.
            for j in range(len(layer)):
                error, next_layer = 0., network[i + 1]
                for neuron in next_layer: # Looping over hidden layer output.
                    error += (neuron['weights'][j] * neuron['delta'])
                errors.append(error)
        else: # Output layer.
            for j in range(len(layer)):
                neuron = layer[j]
                errors.append(expected[j] - neuron['output'])
        for j in range(len(layer)):
            neuron = layer[j]
            gradient = _sigmoid_transfer_derivative(neuron['output'])
            neuron['delta'] = errors[j] * gradient
            if debug:
                neuron['debug_delta'].append(neuron['delta'])
                neuron['debug_error'].append(errors[j])
                neuron['debug_gradient'].append(gradient)

def _update_weights(network, row, l_rate, debug):
    for i in range(len(network)):
        inputs = row[:-1] # All but bias.
        if i != 0:
            prev_layer = network[i - 1] # Looping over hidden layer input.
            inputs = [neuron['output'] for neuron in prev_layer]
        for neuron in network[i]:
            for j in range(len(inputs)):
                neuron['weights'][j] += l_rate * neuron['delta'] * inputs[j]
            neuron['weights'][-1] += l_rate * neuron['delta'] # Bias (associated input = 1.) is the last weight in the list.
            if debug:
                neuron['debug_weights'].append(neuron['weights'])

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
    plt.suptitle('neural network debug_weights')
    for idxl, layer in enumerate(network):
        for idxn, neuron in enumerate(layer):
            axes[idxn][idxl].set_title('layer %s, neuron %s' % (idxl, idxn))
            for idxw in range(len(neuron['weights'])):
                debug_weights = [weights[idxw] for weights in neuron['debug_weights']]
                lbl = 'weights %s' % idxw if idxw != len(neuron['weights']) - 1 else 'bias'
                axes[idxn][idxl].plot(debug_weights, label=lbl)
            axes[idxn][idxl].legend()

def _network_metrics(network, metrics, train_set, val_set):
    _, error = network_evaluate(train_set, network)
    metrics['train_error'].append(error)
    _, error = network_evaluate(val_set, network)
    metrics['val_error'].append(error)

def network_train(train_set, val_set, n_inputs, n_hidden, n_outputs, n_epoch, l_rate, debug=False):
    network = _initialize_network(n_inputs, n_hidden, n_outputs, debug)
    print('network training')
    metrics = {'train_error': [], 'val_error': []}
    for epoch in range(n_epoch):
        sum_error = 0
        for idxr, row in enumerate(train_set):
            outputs = _forward_propagate(network, row)
            expected = [0 for i in range(n_outputs)]
            expected[row[-1]] = 1
            sum_error += _cross_entropy(expected, outputs)
            dbg = True if debug and idxr == len(train_set) - 1 else False
            _backward_propagate_error(network, expected, dbg)
            _update_weights(network, row, l_rate, dbg)
        print('    epoch=%d, lrate=%.3f, sum_error=%.3f' % (epoch, l_rate, sum_error))
        _network_metrics(network, metrics, train_set, val_set)
    if debug:
        _network_debug_plot(network)
    return network, metrics

def network_predict(network, row):
    outputs = _forward_propagate(network, row)
    return outputs.index(max(outputs))

def network_evaluate(data_set, network):
    predictions, good_predictions = [], 0
    for row in data_set:
        expected_row = row[-1]
        predicted_row = network_predict(network, row)
        if predicted_row == expected_row:
            good_predictions += 1
        predictions.append(predicted_row)
    error = (len(data_set) - good_predictions)*100./len(data_set) # Error %
    return predictions, error
