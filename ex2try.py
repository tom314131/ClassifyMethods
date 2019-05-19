import random

import numpy as np

char_to_bin = {"M": [0., 0., 1.], "F": [0., 1., 0.], "I": [1., 0., 0.]}


class Set:
    def __init__(self, samples, labels):
        self.x = samples
        self.y = labels


def load_training_set(x_file_name, y_file_name):
    samples = load_samples(x_file_name)
    normalize_samples(samples)
    labels = load_labels(y_file_name)
    return Set(samples, labels)


def load_test_samples(samples_file_name):
    samples = load_samples(samples_file_name)
    normalize_samples(samples)
    return samples


def load_samples(file_name):
    file = open(file_name)
    data = []
    for line in file:
        splitted_line = line.split(',')
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
    for i in range(3, columns_number):
        column = np.asarray(samples_transpose[i])
        maxs[i] = column.max()
        mins[i] = column.min()

    for row in samples_to_normalize:
        for i in range(3, columns_number):
            if (maxs[i] - mins[i]) != 0:
                row[i] = (row[i] - mins[i]) / (maxs[i] - mins[i])


def normalize_samples(samples_to_normalize):
    # zscore_normalize(samples_to_normalize)
    minmax_normalize(samples_to_normalize)


def shuffle_set(samples, labels):
    shuffled_set = list(zip(samples, labels))
    random.shuffle(shuffled_set)
    return zip(*shuffled_set)


def build_svm(new_set, eta, epochs, gradient,w):
    for e in range(epochs):
        new_set.x, new_set.y = shuffle_set(new_set.x, new_set.y)
        for x, y in zip(new_set.x, new_set.y):
            y_hat = np.argmax(np.dot(w, x))
            if y_hat != y:
                scalar = (1 - eta * gradient)
                w[y] = scalar * w[y] + x * eta
                w[y_hat] = scalar * w[y_hat] - x * eta
                # w[y_hat] = np.subtract(w[y_hat], np.dot(x, eta))
                last_value = 3 - y - y_hat
                w[last_value] = scalar * w[last_value]

    return w

def predict_svm(h, samples,eta,gradient):
    labels = load_labels("new_test_y.txt")
    sample_x, samples_y = shuffle_set(samples, labels)
    count = 0
    error = 0
    for x, y in zip(sample_x, samples_y):
        y_hat = np.argmax(np.dot(h, x))
        if count < 493:
            if y_hat != y:
                scalar = (1 - eta * gradient)
                h[y] = scalar * h[y] + x * eta
                h[y_hat] = scalar * h[y_hat] - x * eta
                # w[y_hat] = np.subtract(w[y_hat], np.dot(x, eta))
                last_value = 3 - y - y_hat
                h[last_value] = scalar * h[last_value]
        if count>=493:
            d = np.argmax(np.dot(h, x))
            if (d != samples_y[count]):
                error = error + 1
        count = count + 1
    return (error/164)

def build_perceptron(new_set, eta, epochs,w):
    for e in range(epochs):
        new_set.x, new_set.y = shuffle_set(new_set.x, new_set.y)
        for x, y in zip(new_set.x, new_set.y):
            y_hat = np.argmax(np.dot(w, x))
            if y_hat != y:
                w[y] = np.add(w[y], np.dot(x, eta))
                w[y_hat] = np.subtract(w[y_hat], np.dot(x, eta))
    return w

def predict_perceptron(h, samples,eta, epochs):
    labels = load_labels("new_test_y.txt")
    sample_x, samples_y = shuffle_set(samples,labels)
    for epoch in range(epochs):
        count = 0
        error = 0
        for x, y in zip(sample_x, samples_y):
            if count< 493:
                y_hat = np.argmax(np.dot(h, x))
                if y_hat != y:
                    h[y] = np.add(h[y], np.dot(x, eta))
                    h[y_hat] = np.subtract(h[y_hat], np.dot(x, eta))

            elif count>=493 and epoch==epochs-1:
                d = np.argmax(np.dot(h, x))
                if(d != samples_y[count]):
                    error = error + 1
            count = count + 1
    return (error/(657-493))
    #for i in range(657):
        #predictions.append(np.argmax(np.dot(h, samples[i])))
    #return np.asarray(predictions)

def calc_test(pred):
    y = load_labels("new_test_y.txt")

    errors = 0

    for i in range(len(pred)):
        if y[i] != pred[i]:
            errors += 1

    return (errors / 657) * 100

def get_ts_and_test(x_file, y_file):
    samples_file = open(x_file)
    labels_file = open(y_file)

    all_samples = np.asarray([line.split() for line in samples_file])
    all_labels = np.asarray([line.split() for line in labels_file])

    mixed_samples, mixed_labels = shuffle_set(all_samples, all_labels)

    test_x_file = open("new_test_x.txt", "w+")
    test_y_file = open("new_test_y.txt", "w+")
    last_index = int(len(mixed_samples) / 5)

    for i in range(0, last_index):
        s1= mixed_samples[i][0] + '\n'
        s2 = mixed_labels[i][0] + '\n'
        test_x_file.write(s1)
        test_y_file.write(s2)
    test_x_file.close()
    test_y_file.close()
    train_x_file = open("new_train_x.txt", "w+")
    train_y_file = open("new_train_y.txt", "w+")
    for i in range(last_index + 1, len(mixed_samples) - 1):
        train_x_file.write(mixed_samples[i][0] + '\n')
        train_y_file.write(mixed_labels[i][0] + '\n')
    train_x_file.write(mixed_samples[len(mixed_samples) - 1][0])
    train_y_file.write(mixed_labels[len(mixed_samples) - 1][0])
    train_x_file.close()
    train_y_file.close()

class Result:

    def __init__(self, eta, epochs, error):
        self.eta = eta
        self.epochs = epochs
        self.error = error

def multi_run():
    training_set = load_training_set("new_train_x.txt", "new_train_y.txt")
    test_samples = load_test_samples("new_test_x.txt")
    errors = []
    for eta in range(1, 11):
        eta = eta * 0.05
        print("checking eta: " + str(eta))
        for epochs in range(1, 5):
            epochs = epochs * 25
            error_avg = 0
            for i in range(10):
                perceptron_hypothesis = build_perceptron(training_set, eta,
                                                         epochs)
                perceptron_prediction = predict_perceptron(
                    perceptron_hypothesis, test_samples)
                error_avg += calc_test(perceptron_prediction)
            errors.append(Result(eta, epochs, error_avg / 10))

    from operator import attrgetter
    errors.sort(key=attrgetter('error'))
    for i in range(10):
        print(
            "eta: " + str(errors[i].eta) + " epochs: " + str(
                errors[i].epochs) + " error: " + str(errors[i].error))

def single_run():
    training_set = load_training_set("new_train_x.txt", "new_train_y.txt")

    test_samples = load_test_samples("new_test_x.txt")
    eta = 0.075
    error_avg = 0
    epochs = 50
    for j in range(10):
        for i in range(5):
            perceptron_hypothesis = np.zeros((3, 10))
            #print("iteration: " + str(i))
            build_perceptron(training_set, eta, epochs, perceptron_hypothesis)
            error_avg += predict_perceptron(perceptron_hypothesis,
                                                       test_samples,eta,epochs)
        print("perceptrone: " + str(error_avg / 5))
        error_avg = 0
        #error_avg += calc_test(perceptron_prediction)


def single_svm():
    training_set = load_training_set("new_train_x.txt", "new_train_y.txt")
    test_samples = load_test_samples("new_test_x.txt")
    eta = 0.05
    error_avg = 0
    epochs = 40
    lambdA= 0.00005
    for j in range(10):
        for i in range(5):
            svm_hypothesis = np.zeros((3, 10))
            #print("iteration: " + str(i))
            build_svm(training_set, eta, epochs,lambdA, svm_hypothesis)
            error_avg+=predict_svm(svm_hypothesis, test_samples, eta, lambdA)
        print("svm: " + str((error_avg / 5)))
        error_avg = 0


if __name__ == '__main__':
    get_ts_and_test("train_x.txt", "train_y.txt")
    single_run()
    single_svm()
    # multi_run()
