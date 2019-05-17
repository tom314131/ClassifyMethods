import csv
import sys
import numpy as np

FEATURE_NUM = 10
SOURCEֹֹ_NUM = 8
def get_data(path):
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
            temp_array.append(1)
            temp_array.append(0)
            temp_array.append(0)

        elif line[0] == 'F':
            temp_array.append(0)
            temp_array.append(1)
            temp_array.append(0)
        else:
            temp_array.append(0)
            temp_array.append(0)
            temp_array.append(1)
        # adding the rest of the catagories
        for i in range(1, SOURCEֹֹ_NUM):
            temp_array.append(float(line[i]))
        # adding the temp_array to the training_array after normalization
        training_array.append(temp_array)
    training_set_file.close()
    return np.array(training_array, float)


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


def training_perceptron(training_set, labels, eta):
    wVectors = np.zeros((3, FEATURE_NUM))
    for k in range(0,100):
        line_number = 0
        for example in training_set:
            max_index = 0
            max_val = vector_mult(example, wVectors[0], FEATURE_NUM)
            for i in range(1, 3):
                temp_val = vector_mult(example, wVectors[i], FEATURE_NUM)
                if max_val < temp_val:
                    max_val = temp_val
                    max_index = i
            correct_class = int(round(labels[line_number][0]))
            if max_index != labels[line_number]:
                wVectors[correct_class] = wVectors[correct_class] + example * eta
                wVectors[max_index] = wVectors[max_index] - example * eta
            line_number = line_number + 1
    return wVectors

def getMaxFeatures(two_dim_array):
    max_features = two_dim_array[0].copy()
    features_num = len(max_features)
    length = len(two_dim_array)
    for array in range(1,length):
        for feature in range(0, features_num):
            if max_features[feature] < two_dim_array[array][feature]:
                max_features[feature] = two_dim_array[array][feature]
    return max_features

def getMinFeatures(two_dim_array):
    min_features = two_dim_array[0].copy()
    features_num = len(min_features)
    length = len(two_dim_array)
    for array in range(1, length):
        for feature in range(0, features_num):
            if min_features[feature] > two_dim_array[array][feature]:
                min_features[feature] = two_dim_array[array][feature]
    return min_features

def normalize_vector(vector, maxFatures, minFeatures):
    temp_arr = []
    for i in range(0,FEATURE_NUM):
        if minFeatures[i] != maxFatures[i]:
            temp_arr.append((vector[i] - minFeatures[i])/(
                maxFatures[i] - minFeatures[i]))
        else :
            temp_arr.append(vector[i])
    return temp_arr


def vector_mult(vector1, vector2, size):
    sum = 0
    for i in range(0, size):
        sum = sum + vector1[i] * vector2[i]
    return sum

def test_perceptron(training_set, labels, weights):
    test_data = get_data(training_set)
    test_label = get_labels(labels)
    size = len(test_data)
    errors = 0
    for i in range(0,size):
        max_val = vector_mult(test_data[i],weights[0],FEATURE_NUM)
        max_index = 0
        for j in range(1,3):
            temp_val = vector_mult(test_data[i],weights[j],FEATURE_NUM)
            if max_val<temp_val:
                max_val = temp_val
                max_index = j
        if max_index != int(round(test_label[i][0])):
          errors= errors +1
    print (100-((errors/size)*100))

def normalize_2dimArray(two_dim_array,maxFeatures,minFeatures):
    length = len(two_dim_array)
    for i in range(0, length):
        two_dim_array[i] = normalize_vector(two_dim_array[i],
                                            maxFeatures,minFeatures)
    return two_dim_array

if __name__ == '__main__':
    # get the training set
    trainingSet = get_data(sys.argv[1])
    maxFeature = getMaxFeatures(trainingSet)
    minFeatures = getMinFeatures(trainingSet)
    normalizeTraining = normalize_2dimArray(trainingSet,maxFeature,minFeatures)
    # print the first row
    #print(len(trainingSet))

    # get the labels
    labels = get_labels(sys.argv[2])
    # print(len(labels))
    # print(perceptron(trainingSet,labels))

    weights = training_perceptron(normalizeTraining, labels,0.1)

    print(weights)
    #test_perceptron(sys.argv[3],sys.argv[4],weights)
