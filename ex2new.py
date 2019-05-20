import random

import numpy as np

char_to_bin = {"M": [0., 0., 1.], "F": [0., 1., 0.], "I": [1., 0., 0.]}


def load_data(x_file_name, y_file_name):
    # load samples
    samples = load_samples(x_file_name)
    normalize_samples(samples)

    # load labels
    labels = load_labels(y_file_name)

    # shuffle and return as set of data
    samples, labels = shuffle_set(samples, labels)
    return samples, labels


def load_test_samples(samples_file_name):
    samples = load_samples(samples_file_name)
    normalize_samples(samples)
    return samples


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


def load_labels(file_name):
    file = open(file_name)
    return np.asarray([int(float(line.split()[0])) for line in file])


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


def minmax_normalize(samples_to_normalize):
    columns_number = len(samples_to_normalize[0])

    samples_transpose = samples_to_normalize.T
    maxs = np.empty([columns_number])
    mins = np.empty([columns_number])
    for i in range(columns_number):
        column = np.asarray(samples_transpose[i])
        maxs[i] = column.max()
        mins[i] = column.min()

    for row in samples_to_normalize:
        for i in range(columns_number):

            if (maxs[i] - mins[i]) != 0:
                row[i] = (row[i] - mins[i]) / (maxs[i] - mins[i])


def normalize_samples(samples_to_normalize):
    # zscore_normalize(samples_to_normalize)
    minmax_normalize(samples_to_normalize)


def shuffle_set(samples, labels):
    shuffled_set = list(zip(samples, labels))
    random.shuffle(shuffled_set)
    return zip(*shuffled_set)


def train_perceptron(data, eta, epochs, w):
    for e in range(epochs):
        random.shuffle(data)

        for x, y in data:
            y_hat = np.argmax(np.dot(w, x))
            if y_hat != y:
                w[y] = np.add(w[y], np.dot(x, eta))
                w[y_hat] = np.subtract(w[y_hat], np.dot(x, eta))

    return w


def train_svm(data, eta, epochs, w, gradient):
    for e in range(epochs):
        random.shuffle(data)

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
            if y_hat != y:
                loss = 1 - np.dot(w[y], x) + np.dot(w[y_hat], x)
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


class Result:

    def __init__(self, eta, epochs, error):
        self.eta = eta
        self.epochs = epochs
        self.error = error


def split_to_equal_arrays(data, parts_num):
    size_of_each_array = len(data) / float(parts_num)
    arrays = []
    last = 0.0

    while last < len(data):
        arrays.append(data[int(last):int(last + size_of_each_array)])
        last += size_of_each_array

    return arrays


def test_hypothesis_on_data(hypothesis, data):
    errors = 0

    for x, y in data:
        y_hay = np.argmax(np.dot(hypothesis, x))
        if y != y_hay:
            errors += 1

    return (errors / len(data)) * 100


def cross_val_perceptron(k_parts, eta, epochs):
    # load and divide data
    samples, labels = load_data("train_x.txt", "train_y.txt")
    data = list(zip(samples, labels))
    arrays = split_to_equal_arrays(data, k_parts)

    # create initial hypothesis
    features_number = len(samples[0])
    classes_number = len(set(labels))
    hypothesis = np.zeros((classes_number, features_number))

    # train and validate using cross validation with k parts
    errors = 0
    for i in range(k_parts):
        for j in (value for value in range(k_parts) if value != i):
            hypothesis = train_perceptron(arrays[j], eta, epochs, hypothesis)
        errors += test_hypothesis_on_data(hypothesis, arrays[i])

    return errors / k_parts


def cross_val_svm(k_parts, eta, epochs, gradient):
    # load and divide data
    samples, labels = load_data("train_x.txt", "train_y.txt")
    data = list(zip(samples, labels))
    arrays = split_to_equal_arrays(data, k_parts)

    # create initial hypothesis
    features_number = len(samples[0])
    classes_number = len(set(labels))
    hypothesis = np.zeros((classes_number, features_number))

    # train and validate using cross validation with k parts
    errors = 0
    for i in range(k_parts):
        for j in (value for value in range(k_parts) if value != i):
            hypothesis = train_svm(arrays[j], eta, epochs, hypothesis, gradient)
        errors += test_hypothesis_on_data(hypothesis, arrays[i])

    return errors / k_parts


def cross_val_pa(k_parts, epochs):
    # load and divide data
    samples, labels = load_data("train_x.txt", "train_y.txt")
    data = list(zip(samples, labels))
    arrays = split_to_equal_arrays(data, k_parts)

    # create initial hypothesis
    features_number = len(samples[0])
    classes_number = len(set(labels))
    hypothesis = np.zeros((classes_number, features_number))

    # train and validate using cross validation with k parts
    errors = 0
    for i in range(k_parts):
        for j in (value for value in range(k_parts) if value != i):
            hypothesis = train_pa(arrays[j], epochs, hypothesis)
        errors += test_hypothesis_on_data(hypothesis, arrays[i])

    return errors / k_parts


def test_cross_val_with_params():
    random.seed(30)
    errors = []

    for eta in range(1, 10):
        eta = eta / 100
        print("checking eta ", eta, " with epoch: ", end=" ")
        for epoch in range(10, 16):
            print(epoch, end=" - ")
            error = cross_val_perceptron(5, eta, epoch)
            print(str(error)[:5], end="%, ")
            errors.append(Result(eta, epoch, error))
        print()

    from operator import attrgetter
    errors.sort(key=attrgetter('error'))
    for i in range(min(len(errors), 10)):
        print(
            "eta: " + str(errors[i].eta) + " epochs: " + str(errors[i].epochs) + " error: " + str(errors[i].error))


if __name__ == '__main__':
    # generate_ts_and_test_from_files("train_x.txt", "train_y.txt")
    # single_run()
    # test_cross_val_with_params()
    random.seed(30)
    print(cross_val_perceptron(5, 0.07, 19))

    # arr = split_to_equal_arrays(data, 3)
    # s, l = zip(*arr[0])
    # print(data[0])
    # pa_tests()
    # tests()
