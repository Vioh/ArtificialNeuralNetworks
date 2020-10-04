import time
import numpy as np

# Global parameters
eta = 0.02
max_number_of_updates = int(1e5)
max_number_of_repetitions = 10

# Loading data
X = np.genfromtxt("input_data_numeric.csv", delimiter=",")[:,1:]
Y = {
    "A": np.array([+1, +1, -1, +1, -1, +1, +1, +1, -1, -1, -1, -1, -1, -1, -1, -1]),
    "B": np.array([+1, -1, +1, +1, +1, -1, +1, +1, +1, +1, +1, +1, -1, +1, +1, +1]),
    "C": np.array([-1, -1, -1, +1, -1, -1, +1, -1, +1, -1, +1, +1, -1, +1, -1, +1]),
    "D": np.array([+1, -1, +1, -1, -1, +1, -1, -1, -1, -1, -1, -1, +1, -1, -1, +1]),
    "E": np.array([+1, +1, +1, +1, -1, -1, -1, -1, -1, +1, -1, -1, +1, -1, -1, -1]),
    "F": np.array([-1, -1, +1, +1, +1, -1, -1, -1, -1, +1, +1, -1, +1, +1, +1, -1]),
}

def check_linear_separability(function_name):
    start_time = time.time()
    y_true = Y[function_name]
    linearly_separable = False

    for i_repetition in range(1, max_number_of_repetitions + 1):
        W = np.random.uniform(low=-0.2, high=0.2, size=(4,))
        theta = np.random.uniform(low=-1.0, high=1.0, size=None)

        for i_update in range(1, max_number_of_updates + 1):
            # Forward propagate for all data points
            local_fields = (X @ W - theta) / 2
            outputs = np.tanh(local_fields)
            y_pred = np.sign(outputs)

            # Check convergence criteria using all data points
            if np.array_equal(y_pred, y_true):
                linearly_separable = True
                break

            # Backward propagate to update weights and threshold
            mu = np.random.choice(X.shape[0])
            error = (y_true[mu] - outputs[mu]) * (1 - np.tanh(local_fields[mu])**2)
            W += eta * error * X[mu]
            theta -= eta * error

        if linearly_separable:
            break

    print("Results for Boolean function " + function_name + ":")
    print("    Linearly separable:", "YES" if linearly_separable else "NO")
    print("    Final weights     :", W)
    print("    Final threshold   :", theta)
    print("    Total time taken  :", (time.time() - start_time), "seconds")
    print("    Total #repetitions:", i_repetition)
    print("    Total #updates in last repetition: ", i_update)

if __name__ == "__main__":
    [check_linear_separability(name) for name in Y.keys()]
