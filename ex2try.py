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


def normalize_samples(samples_to_normalize):
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


def build_perceptron(new_set):
    x = new_set.x
    y = new_set.y
    eta = 0.1
    samples_number = len(x)
    features_number = len(x[0])
    labels_number = 3
    w = np.zeros(labels_number, features_number)

    for i in range(samples_number):
        result_vector = np.dot(w, x[i])
        y_hat = (np.where(result_vector == np.amax(result_vector)))[0][0]
        if y_hat != y[i]:
            w[y[i]] = np.add(w[y[i]], np.dot(x, eta))
            w[y_hat] = np.subtract(w[y_hat], np.dot(x, eta))

    return w


def predict_perceptron(h, samples):
    samples_number = len(samples)
    predictions = np.array([samples_number])
    for i in range(samples_number):
        result_vector = np.dot(h, samples[i])
        predictions[i] = (np.where(result_vector == np.amax(result_vector)))[0][0]
    return predictions


if __name__ == '__main__':
    training_set = load_training_set("train_x_small.txt", "train_y_small.txt")

    test_samples = load_test_samples("text_x.txt")

    perceptron_hypothesis = build_perceptron(training_set)
    perceptron_prediction = predict_perceptron(perceptron_hypothesis, test_samples)
