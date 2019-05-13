import csv
import sys
import numpy as np


def get_training_set(path):
    # open the file
    training_set_file = open(path)
    # creating the array of the training set
    training_array = []
    line_number = 0
    csv_training = csv.reader(training_set_file)
    for line in csv_training:
        # creating a temp array for the current row
        temp_array = []
        # adding the sex type---------------------------------
        if line[0] == 'M':
            temp_array.append(0)
        elif line[0] == 'F':
            temp_array.append(1)
        else:
            temp_array.append(2)
        # adding the rest of the catagories
        for i in range(1, 8):
            temp_array.append(float(line[i]))
        # adding the temp_array to the training_array
        training_array.append(temp_array)
    training_set_file.close()
    return np.array(training_array,float)


def get_labels(path):
    # open the file
    labels_file = open(path)
    # creating the array of the labels
    labels_array = []
    csv_labels = csv.reader(labels_file)
    for line in csv_labels:
        # adding line to labels_array
        labels_array.append(line)
    labels_file.close()
    # return array with float values
    return np.array(labels_array, float)

def perceptron(training_set, labels):
    eta = 0.1
    vector_size = len(training_set[0])
    w = np.zeros(vector_size,)
    for i in range(0,len(labels)):
        # training_set[i][0]=0
        u = 0
        for j in range(0,vector_size):
            u = u + w[j] * training_set[i][j]
        if labels[i]*u<=0:
            w = w + eta*labels[i]*trainingSet[i]
    perceptron_w = w
    return perceptron_w


if __name__ == '__main__':
    # get the training set
    trainingSet = get_training_set(sys.argv[1])
    # print the first row
    #print(trainingSet[0])

    # get the labels
    labels = get_labels(sys.argv[2])
    #print(len(labels))
    print(perceptron(trainingSet,labels))
