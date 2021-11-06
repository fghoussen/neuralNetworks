#!/usr/bin/python
# -*- coding: UTF-8 -*-

from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
import neural_network_library as nnl
import matplotlib.pyplot as plt

from random import seed
seed(0) # Get repeatable results.

cm = plt.cm.get_cmap('rainbow')

def main():
    # Create random data.
    n_features, n_targets = 1, 1
    bias, noise = 5., 5. # Spread data to make them less/more predictable/noisy.
    x, y = make_regression(n_samples=250, n_features=n_features, n_targets=n_targets, bias=bias, noise=noise,
                           random_state=0, n_informative=n_features, shuffle=True)
    _, axes = plt.subplots(3, 6)
    plt.suptitle('neural network results')
    axes[0][0].scatter(x[:], y[:], c=y, vmin=min(y), vmax=max(y), cmap=cm)
    axes[0][0].set_title('data')
    xlim, ylim = axes[0][0].get_xlim(), axes[0][0].get_ylim()

    # Scale data to reduce weights.
    classification = False # Regression.
    nnl.network_preprocess_data('minmax_std', x, y, classification)
    x_scaled = nnl.SCALER_PIPELINE_X.transform(x).reshape(x.shape)
    y_scaled = nnl.SCALER_PIPELINE_Y.transform(y.reshape(len(y), 1)).reshape(y.shape)
    axes[0][1].scatter(x_scaled[:], y_scaled[:], c=y_scaled, vmin=min(y_scaled), vmax=max(y_scaled), cmap=cm)
    axes[0][1].set_title('scaled data (zoom unchanged)')
    axes[0][1].set_xlim(xlim), axes[0][1].set_ylim(ylim)
    axes[0][2].scatter(x_scaled[:], y_scaled[:], c=y_scaled, vmin=min(y_scaled), vmax=max(y_scaled), cmap=cm)
    axes[0][2].set_title('scaled data (zoom fit)')

    # Split data set into training set, validation set and testing set.
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.25)

    # Train and test neural network.
    n_outputs, n_hidden = 1, 1 # Regression: only one output.
    n_epoch, alpha = 200, 0.001
    for idxl, beta1, beta2, batch_size in zip([1, 2], [None, 0.9], [None, 0.999], [4, 4]):
        for idxc, activation_function in zip([0, 3], ['sigmoid', 'relu']):
            # Train neural network.
            train_set = [[x_train[i, 0], y_train[i]] for i in range(x_train.shape[0])]
            val_set = [[x_val[i, 0], y_val[i]] for i in range(x_val.shape[0])]
            network, metrics = nnl.network_train(train_set, val_set, classification,
                                                 n_hidden, activation_function, n_outputs, 'linear',
                                                 n_epoch, alpha, beta1, beta2,
                                                 batch_size=batch_size, debug=False)
            predictions, error = nnl.network_evaluate(train_set, network, classification)
            if idxc == 0:
                method = 'Adam' if beta1 is not None and beta2 is not None else 'SGD' # Adam or SGD
                axes[idxl][idxc+0].text(-0.5, 0.5, method, transform=axes[idxl][idxc+0].transAxes)
            axes[idxl][idxc+0].plot(metrics['train_error'], label='train error', marker='o')
            axes[idxl][idxc+0].plot(metrics['val_error'], label='val error', marker='o')
            axes[idxl][idxc+0].set_xlabel('epoch')
            axes[idxl][idxc+0].set_ylabel('error (%)')
            axes[idxl][idxc+0].set_title('learning curve %s' % activation_function)
            axes[idxl][idxc+0].legend()
            axes[idxl][idxc+1].scatter([ts[0] for ts in train_set], predictions, c=predictions,
                                       vmin=min(predictions), vmax=max(predictions), cmap=cm)
            axes[idxl][idxc+1].set_title('train %s - error %.2f %%' % (activation_function, error))

            # Test neural network.
            test_set = [[x_test[i, 0], y_test[i]] for i in range(x_test.shape[0])]
            predictions, error = nnl.network_evaluate(test_set, network, classification)
            axes[idxl][idxc+2].scatter([ts[0] for ts in test_set], predictions, c=predictions,
                                       vmin=min(predictions), vmax=max(predictions), cmap=cm)
            axes[idxl][idxc+2].set_title('test %s - error %.2f %%' % (activation_function, error))

    # Show Results.
    plt.subplots_adjust(left=0.1, bottom=0.05, right=0.9, top=0.9, wspace=0.3, hspace=0.3)
    plt.show()

if __name__ == '__main__':
    main()
