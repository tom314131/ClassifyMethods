def pa_tests():
    training_set = load_data("new_train_x.txt", "new_train_y.txt")
    features_number = len(training_set.x[0])
    classes_number = len(set(training_set.y))

    test_samples = load_test_samples("new_test_x.txt")
    errors = []

    for epochs in range(1, 10):
        epochs = epochs * 10
        print("checking epoch: ", epochs)
        error_avg = 0
        for i in range(100):
            hypothesis = build_pa(training_set, epochs, features_number, classes_number)
            perceptron_prediction = predict_using_hypothesis(hypothesis, test_samples)
            error_avg += check_predictions(perceptron_prediction)
        errors.append(Result(0, epochs, error_avg / 100))

    from operator import attrgetter

    errors.sort(key=attrgetter('error'))
    for i in range(min(len(errors), 10)):
        print("epochs: " + str(errors[i].epochs) + " error: " + str(errors[i].error))


def multi_run():
    training_set = load_data("new_train_x.txt", "new_train_y.txt")
    features_number = len(training_set.x[0])
    classes_number = len(set(training_set.y))

    test_samples = load_test_samples("new_test_x.txt")
    errors = []

    for eta in range(100):
        eta = eta * 0.05
        print("checking eta: " + str(eta))
        for epochs in range(1, 10):
            epochs = epochs * 10
            print("    checking epoch: " + str(epochs))
            error_avg = 0
            for i in range(10):
                hypothesis = build_pa(training_set, epochs, features_number, classes_number)
                perceptron_prediction = predict_using_hypothesis(hypothesis, test_samples)
                error_avg += check_predictions(perceptron_prediction)
            errors.append(Result(eta, epochs, error_avg / 10))

    from operator import attrgetter
    errors.sort(key=attrgetter('error'))
    for i in range(min(len(errors), 10)):
        print(
            "eta: " + str(errors[i].eta) + " epochs: " + str(errors[i].epochs) + " error: " + str(errors[i].error))


def check_predictions(predictions):
    y = load_labels("new_test_y.txt")
    number_of_predictions = len(predictions)

    errors = 0

    for i in range(number_of_predictions):
        if y[i] != predictions[i]:
            errors += 1

    return (errors / number_of_predictions) * 100


# def single_run():
#     training_set = load_data("new_train_x.txt", "new_train_y.txt")
#     features_number = len(training_set.x[0])
#     classes_number = len(set(training_set.y))
#
#     test_samples = load_test_samples("new_test_x.txt")
#     error_avg = 0
#     error_num = 0
#
#     for i in range(100):
#         perceptron_hypothesis = build_pa(training_set, 10, features_number, classes_number)
#         # perceptron_hypothesis = build_perceptron(training_set, 0.1, 10, features_number, classes_number)
#         perceptron_prediction = predict_using_hypothesis(perceptron_hypothesis, test_samples)
#         error = check_predictions(perceptron_prediction)
#         error_avg += error
#         error_num += 1
#         print(error)
#
#     error_avg /= error_num
#     print("for ", error_num, " average is ", error_avg)

def generate_ts_and_test_from_files(x_file, y_file):
    # load files
    samples_file = open(x_file)
    labels_file = open(y_file)

    all_samples = np.asarray([line.split() for line in samples_file])
    all_labels = np.asarray([line.split() for line in labels_file])

    # mix data
    mixed_samples, mixed_labels = shuffle_set(all_samples, all_labels)

    # create test files
    test_x_file = open("new_test_x.txt", "w+")
    test_y_file = open("new_test_y.txt", "w+")
    last_index = int(len(mixed_samples) / 5)

    for i in range(0, last_index - 1):
        test_x_file.write(mixed_samples[i][0] + '\n')
        test_y_file.write(mixed_labels[i][0] + '\n')
    test_x_file.write(mixed_samples[last_index - 1][0])
    test_y_file.write(mixed_labels[last_index - 1][0])

    # create train files
    train_x_file = open("new_train_x.txt", "w+")
    train_y_file = open("new_train_y.txt", "w+")
    for i in range(last_index, len(mixed_samples) - 1):
        train_x_file.write(mixed_samples[i][0] + '\n')
        train_y_file.write(mixed_labels[i][0] + '\n')
    train_x_file.write(mixed_samples[len(mixed_samples) - 1][0])
    train_y_file.write(mixed_labels[len(mixed_samples) - 1][0])
