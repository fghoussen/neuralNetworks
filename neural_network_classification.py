#!/usr/bin/python
# -*- coding: UTF-8 -*-

from sklearn.datasets import make_blobs
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from neural_network_library import network_train, network_predict, network_evaluate
import matplotlib.pyplot as plt

from random import seed
seed(0) # Get repeatable results.

def main():
    # Create random data.
    n_classes = 3
    std_classes = 2 # Spread classes around centers to make data less/more predictable/noisy.
    x, y = make_blobs(n_samples=1000, centers=n_classes, random_state=0, cluster_std=std_classes)
    _, axes = plt.subplots(2, 5)
    plt.suptitle('neural network results')
    axes[0][0].scatter(x[:, 0], x[:, 1], c=y)
    axes[0][0].set_title('data')
    xlim, ylim = axes[0][0].get_xlim(), axes[0][0].get_ylim()

    # Scale data to reduce weights.
    std_scale = preprocessing.StandardScaler().fit(x)
    x_scaled = std_scale.transform(x)
    axes[0][1].scatter(x_scaled[:, 0], x_scaled[:, 1], c=y)
    axes[0][1].set_title('scaled data (zoom unchanged)')
    axes[0][1].set_xlim(xlim), axes[0][1].set_ylim(ylim)
    axes[1][1].scatter(x_scaled[:, 0], x_scaled[:, 1], c=y)
    axes[1][1].set_title('scaled data (zoom fit)')
    xlim_scaled, ylim_scaled = axes[1][1].get_xlim(), axes[1][1].get_ylim()

    # Split data set into training set, validation set and testing set.
    x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.25)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.25)

    # Train neural network.
    train_set = [[x_train[i, 0], x_train[i, 1], y_train[i]] for i in range(x_train.shape[0])]
    val_set = [[x_val[i, 0], x_val[i, 1], y_val[i]] for i in range(x_val.shape[0])]
    network, metrics = network_train(train_set, val_set,
                                     2, 2, n_classes, 'sigmoid',
                                     1000, 0.001, debug=True)
    predictions, error = network_evaluate(train_set, network)
    axes[1][2].plot(metrics['train_error'], label='train error', marker='o')
    axes[1][2].plot(metrics['val_error'], label='val error', marker='o')
    axes[1][2].set_xlabel('epoch')
    axes[1][2].set_title('learning curve SGD - error (%)')
    axes[1][2].legend()
    axes[1][3].scatter([ts[0] for ts in train_set], [ts[1] for ts in train_set], c=predictions)
    axes[1][3].set_title('train SGD - error %.2f %%' % error)
    axes[1][3].set_xlim(xlim_scaled), axes[1][3].set_ylim(ylim_scaled)

    # Test neural network.
    test_set = [[x_test[i, 0], x_test[i, 1], y_test[i]] for i in range(x_test.shape[0])]
    predictions, error = network_evaluate(test_set, network)
    axes[1][4].scatter([ts[0] for ts in test_set], [ts[1] for ts in test_set], c=predictions)
    axes[1][4].set_title('test SGD - error %.2f %%' % error)
    axes[1][4].set_xlim(xlim_scaled), axes[1][4].set_ylim(ylim_scaled)

    # Show Results.
    plt.show()

if __name__ == '__main__':
    main()
