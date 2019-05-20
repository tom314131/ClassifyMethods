import random

import numpy as np

# load and data section
char_to_bin = {"M": [0., 0., 1.], "F": [0., 1., 0.], "I": [1., 0., 0.]}


class TrainingSet:

    def __init__(self, samples, labels, mins, maxs):
        self.features_number = len(samples[0])
        self.classes_number = len(set(labels))
        self.data = list(zip(samples, labels))
        self.mins = mins
        self.maxs = maxs

    def shuffle_data(self):
        random.shuffle(self.data)
        return self


def load_training_set(x_file_name, y_file_name):
    # load samples
    mins = []
    maxs = []
    samples = load_samples(x_file_name)
    normalize_samples(samples, mins, maxs)

    # load labels
    labels = load_labels(y_file_name)

    # shuffle and return as set of data
    training_set = TrainingSet(samples, labels, mins, maxs)

    return training_set.shuffle_data()


def load_samples(file_name):
    file = open(file_name)
    data = []
    for line in file:
        splitted_line = (line.split()[0]).split(',')
        ts = char_to_bin[splitted_line[0]].copy()
        ts.extend(float(splitted_line[i]) for i in range(1, len(splitted_line)))
        data.append(ts)
    file.close()
    return np.asarray(data)


def zscore_normalize(samples_to_normalize):
    columns_number = len(samples_to_normalize[0])

    samples_transpose = samples_to_normalize.T
    means = np.empty([columns_number])
    deviations = np.empty([columns_number])

    for i in range(3, columns_number):
        column = np.asarray(samples_transpose[i])
        means[i] = column.mean()
        deviations[i] = column.std()

    for row in samples_to_normalize:
        for i in range(3, columns_number):
            if deviations[i] != 0:
                row[i] = (row[i] - means[i]) / deviations[i]


def minmax_normalize(samples_to_normalize, mins, maxs):
    columns_number = len(samples_to_normalize[0])

    if len(mins) == 0:
        samples_transpose = samples_to_normalize.T
        mins = np.empty([columns_number])
        maxs = np.empty([columns_number])
        for i in range(columns_number):
            column = np.asarray(samples_transpose[i])
            maxs[i] = column.max()
            mins[i] = column.min()

    for row in samples_to_normalize:
        for i in range(columns_number):
            if (maxs[i] - mins[i]) != 0:
                row[i] = (row[i] - mins[i]) / (maxs[i] - mins[i])


def normalize_samples(samples_to_normalize, mins=None, maxs=None):
    # zscore_normalize(samples_to_normalize)
    minmax_normalize(samples_to_normalize, mins, maxs)


def load_labels(file_name):
    file = open(file_name)
    labels = np.asarray([int(float(line.split()[0])) for line in file])
    file.close()
    return labels


def load_test_samples(samples_file_name, mins=None, maxs=None):
    if mins is None:
        mins = []
    if maxs is None:
        maxs = []
    samples = load_samples(samples_file_name)
    normalize_samples(samples, mins, maxs)

    return samples


# train and predict  section

def train_perceptron(data, eta, epochs, w):
    for e in range(epochs):
        random.shuffle(data)
        eta = np.power(eta, 2)

        for x, y in data:
            y_hat = np.argmax(np.dot(w, x))
            if y_hat != y:
                w[y] = np.add(w[y], np.dot(x, eta))
                w[y_hat] = np.subtract(w[y_hat], np.dot(x, eta))

    return w


def train_svm(data, eta, epochs, w, gradient):
    for e in range(epochs):
        random.shuffle(data)
        eta = np.power(eta, 2)

        for x, y in data:
            y_hat = np.argmax(np.dot(w, x))
            if y_hat != y:
                scalar = (1 - eta * gradient)
                w[y] = scalar * w[y] + x * eta
                w[y_hat] = scalar * w[y_hat] - x * eta
                last_value = 3 - y - y_hat
                w[last_value] = scalar * w[last_value]

    return w


def train_pa(data, epochs, w):
    for e in range(epochs):
        random.shuffle(data)

        for x, y in data:
            y_hat = np.argmax(np.dot(w, x))
            loss = max(0, 1 - np.dot(w[y], x) + np.dot(w[y_hat], x))
            tau = loss / (2 * (np.power((np.linalg.norm(x)), 2)))
            w[y] = w[y] + (tau * x)
            w[y_hat] = w[y_hat] - (tau * x)

    return w


def predict_using_hypothesis(hypothesis, samples):
    predictions = []
    number_of_samples = len(samples)

    for i in range(number_of_samples):
        predictions.append(np.argmax(np.dot(hypothesis, samples[i])))

    return np.asarray(predictions)


if __name__ == '__main__':
    import sys

    args = sys.argv

    # load files
    training_set = load_training_set(args[1], args[2])
    test_samples = load_test_samples(args[3], training_set.mins, training_set.maxs)

    # get perceptrons answers
    hypothesis = np.zeros((training_set.classes_number, training_set.features_number))
    hypothesis = train_perceptron(training_set.data, 0.7, 44, hypothesis)
    perceptron_results = predict_using_hypothesis(hypothesis, test_samples)

    # get svm answers
    hypothesis = np.zeros((training_set.classes_number, training_set.features_number))
    hypothesis = train_svm(training_set.data, 0.7, 20, hypothesis, 0.00075)
    svm_results = predict_using_hypothesis(hypothesis, test_samples)

    # get pa answers
    hypothesis = np.zeros((training_set.classes_number, training_set.features_number))
    hypothesis = train_pa(training_set.data, 85, hypothesis)
    pa_results = predict_using_hypothesis(hypothesis, test_samples)

    # print answers
    for i in range(len(test_samples)):
        print("perceptron: {}, svm: {}, pa: {}".format(perceptron_results[i], svm_results[i], pa_results[i]))
