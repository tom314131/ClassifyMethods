import numpy as np

char_to_bin = {"M": [0., 0., 1.], "F": [0., 1., 0.], "I": [1., 0., 0.]}


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


if __name__ == '__main__':
    samples = load_samples("train_x_small.txt")
    normalize_samples(samples)

    labels = load_labels("train_y_small.txt")
