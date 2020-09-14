import time
import random
import argparse
import numpy as np
from progressbar import progressbar # pip install progressbar2


def noisy_sigmoid(value, beta=2):
    """
    Compute the noisy sigmoid activation in stochastic Hopfield network, with noise parameter beta.
    """
    return 1 / (1 + np.exp(-2 * beta * value))


def generate_random_patterns(p, N=200):
    """
    Generate `p` random patterns, each with `N` bits/neurons.
    """
    patterns = np.random.randint(2, size=(p, N))
    patterns[patterns == 0] = -1
    return patterns


def compute_weight_matrix(stored_patterns, row=None, zerodiag=True):
    """
    Create weight matrix for Hopfield network using Hebb's rule.

    Args:
        stored_patterns (ndarray): Tensor of shape `(p, N)` containing `p` stored patterns, each
            represented by `N` neurons/bits.

        row (int): Create weight matrix only for this row, such that the return value will have the
            shape of `(1, N)` (where N is the number of bits in each stored pattern). If no row is
            given, then the entire weight matrix of shape `(N, N)` will be computed.

        zerodiag (bool): If true, the weights along the diagonal of the weight matrix will be zeroed
            out. Otherwise, the diagonal weights will be computed using the normal Hebb's rule.

    Returns:
        The weight matrix of shape `(1, N)` if row is given. Otherwise, the entire matrix of shape
        `(N, N)` will be returned.
    """
    N = stored_patterns.shape[1]
    W = stored_patterns if row is None else stored_patterns[:, row, None]
    W = (W.T @ stored_patterns) / N

    if zerodiag:
        if row is None:
            np.fill_diagonal(W, 0)
        else:
            W[:,row] = 0
    return W


def compute_order_parameter(input_pattern, weight_matrix, T=int(2e5)):
    """
    Run the stochastic Hopfield network to compute the order parameter.

    Args:
        input_pattern (ndarray): Tensor of shape `(N,)` representing the input pattern with N bits.

        weight_matrix (ndarray): Tensor of shape `(N, N)` representing weight matrix of the network.

        T (int): Number of asynchronous updates used for computing the order parameter.

    Returns:
        A number representing the order parameter of the stochastic Hopfield network.
    """
    order = 0
    pattern = np.copy(input_pattern)
    N = len(pattern)

    for _ in range(T):
        random_neuron = random.randrange(N)
        prob = noisy_sigmoid(np.dot(weight_matrix[random_neuron], pattern))
        pattern[random_neuron] = 1 if random.random() <= prob else -1
        order += np.dot(pattern, input_pattern) / N / T

    return order


def run_experiments(p, n_trials=100):
    """
    Run all the experiments to compute the average order parameter.
    """
    orders = []
    start_time = time.time()

    for _ in progressbar(range(n_trials)):
        random_patterns = generate_random_patterns(p)
        W = compute_weight_matrix(random_patterns)
        orders.append(compute_order_parameter(random_patterns[0], W))

    print("Average order parameter: {:.3f}".format(sum(orders) / len(orders)))
    print("Total time taken: {} seconds".format(time.time() - start_time))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute order parameter for stochastic Hopfield network"
    )
    parser.add_argument(
        "p",
        type=int,
        help="Number of stored patterns"
    )
    args = parser.parse_args()
    run_experiments(args.p)
