import argparse
import numpy as np
import matplotlib.pyplot as plt

# Load training data
train_set = np.genfromtxt("training_set.csv", delimiter=",").T
n_train = train_set.shape[1]

# Load validation data
val_set = np.genfromtxt("validation_set.csv", delimiter=",").T
n_val = val_set.shape[1]
X_val = val_set[:-1, :]
Y_val = val_set[-1:, :]
input_dim = X_val.shape[0]

# Pre-defined constants
accepted_error_rate = 0.12
eta = 0.02
n_epochs = 1000
n_layers = 3
n_dims = [input_dim, 8, 16, 1] # number of dimensions in all layers (including input layer)

# Initialize weights and thresholds for all layers
W = [np.random.normal(0, 1, size=(n_dims[l], n_dims[l-1])) for l in range(1, n_layers+1)]
T = [np.zeros(shape=(n_dims[l], 1)) for l in range (1, n_layers+1)]


def tanh_derivative(values):
    return (1 - np.tanh(values)**2)


def run_perceptron(inputs, targets, is_training=False):
    X = [inputs] # store inputs for all layers
    B = []       # store local fields for all layers
    O = []       # store outputs for all layers
    E = []       # store errors for all layers

    # Forward propagate
    for l in range(n_layers):
        B.append(W[l] @ X[-1] - T[l])
        O.append(np.tanh(B[-1]))

        if l != n_layers - 1:
            X.append(O[-1])

    # Backward propagate only during training (assuming batch size of 1)
    if is_training and inputs.shape[1] == 1:
        E.insert(0, ((targets - O[-1]) * tanh_derivative(B[-1])))

        for l in range(-1, -n_layers, -1):
            E.insert(0, (W[l].T @ E[0] * tanh_derivative(B[l-1])))

        for l in range(n_layers):
            W[l] += eta * E[l] @ X[l].T
            T[l] -= eta * E[l]

    return O[-1]


def main(args):
    np.random.seed(args.seed)

    for epoch in range(1, n_epochs + 1):
        np.random.shuffle(train_set.T)
        X_train = train_set[:-1, :]
        Y_train = train_set[-1:, :]

        # Training
        for mu in range(n_train):
            run_perceptron(X_train[:,mu,None], Y_train[:,mu,None], is_training=True)

        # Validating
        Y_pred = np.sign(run_perceptron(X_val, Y_val, is_training=False))
        error_rate = (Y_pred != Y_val).sum() / n_val

        # Print summary and check stopping criteria
        print("[Epoch " + str(epoch) + "]", "Error rate:", error_rate)
        if error_rate <= accepted_error_rate: break

    for l in range(n_layers):
        np.savetxt("{}/w{}.csv".format(args.outdir, l+1), W[l], delimiter=",")
        np.savetxt("{}/t{}.csv".format(args.outdir, l+1), T[l], delimiter=",")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Experiment with 2-layer perceptron")
    parser.add_argument("--outdir", "-o", type=str, default=".", help="Out directory")
    parser.add_argument("--seed", "-s", type=int, default=None, help="Seed for random generator")
    main(parser.parse_args())
