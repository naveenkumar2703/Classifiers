

import numpy as np


class LogisticClassifier:
    def __init__(self, number_of_features, labels):
        self.number_of_features = number_of_features
        self.number_of_labels = len(labels)
        self.labels = list(labels)
        # -1 to number of labels as weight is not needed for last label and can be computed from probability axioms
        #  + 1 to number of features to hold a dummy x for bias weight.
        self.weights = np.zeros(((self.number_of_labels - 1), (number_of_features + 1)))


    def compute_exponential_product(self, data, weight):
        data_fmt = np.array(data).astype(float)
        data_fmt = np.append(data_fmt, 1.0) # adding 1 for bias - weight
        dot_prod = np.dot(weight, data_fmt)
        return np.exp(dot_prod)
    # eta - learning rate
    # lamda - regularization factor
    def train(self, training_data, training_labels, eta, lamda):
        #print('Computing weights')
        eta_by_lambda = eta * lamda
        for index in range(len(training_data)):
            data = training_data[index]
            current_label = training_labels[index]
            exponential_product_of_label = []
            for label_index in range(self.number_of_labels - 1):  # computing for k-1 labels
                exponential_product_of_label.append(
                    self.compute_exponential_product(data, self.weights[label_index]))

            sum_exponential_product_of_label = sum(exponential_product_of_label)  # sum values of k-1 labels
            est_probability_of_label = np.array(exponential_product_of_label)
            est_probability_of_label = est_probability_of_label / (
                    1 + sum_exponential_product_of_label)  # computing k - 1 probabilities

            for label_index in range(len(est_probability_of_label)):
                label = self.labels[label_index]
                expected_value = 0
                if current_label == label:
                    expected_value = 1

                # computing error
                delta = expected_value - est_probability_of_label[label_index]
                data_fmt = np.array(data).astype(float)
                data_fmt = np.append(data_fmt, 1)
                delta_data = delta * data_fmt
                # updating weights with error based on learning rate and regularization factor
                self.weights[label_index] = self.weights[label_index] + (eta * delta_data) + (eta_by_lambda) * self.weights[label_index]





    def predict(self, test_data):
        exponential_product_of_label = []
        for label_index in range(self.number_of_labels - 1): # computing for k-1 labels
            exponential_product_of_label.append(self.compute_exponential_product(test_data, self.weights[label_index]))
        sum_exponential_product_of_label = sum(exponential_product_of_label) # sum values of k-1 labels
        probability_of_label = [item / (1 + sum_exponential_product_of_label) for item in exponential_product_of_label]
        #probability_of_label = probability_of_label/(1 + sum_exponential_product_of_label) # computing k - 1 probabilities
        probability_of_label.append((1 - sum_exponential_product_of_label)) # computing probability of last (kth) label.
        max_probability_index = probability_of_label.index(max(probability_of_label))
        return self.labels[max_probability_index]

# Takes input as array of training examples and array of labels
#and returns a list of weights as output.
def build_classifier(data, labels, learning_rate, regularization_factor, max_iterations, exit_accuracy, cv):
    label_values = set(labels)
    number_of_labels = len(label_values)
    number_of_features = len(data[0]) # accessing first data item to find out number of attributes
    all_data_size = len(data)
    test_data_size = int(len(data) / cv)
    train_data_size = all_data_size - test_data_size
    accuracies = []
    classifiers = []
    for index in range(cv):
        train_start_index = 0
        train_end_index = all_data_size

        train_first_splice = (train_data_size - index * test_data_size)
        train_second_splice = (all_data_size - index * test_data_size)
        if index == (cv-1):# end index
            train_start_index = test_data_size
            train_first_splice = all_data_size
            train_second_splice = all_data_size
        elif index == 0:# start index
            train_end_index = all_data_size
            train_second_splice = all_data_size

        train_data = data[train_start_index:train_first_splice + 1]
        train_data.extend(data[train_second_splice:train_end_index])
        train_labels = labels[train_start_index:train_first_splice + 1]
        train_labels.extend(labels[train_second_splice:train_end_index])
        test_data = data[(train_data_size - index * test_data_size):(all_data_size - index * test_data_size)]
        test_labels = labels[(train_data_size - index * test_data_size):(all_data_size - index * test_data_size)]
        classifier = LogisticClassifier(number_of_features, label_values)
        correct_classification = 0
        iter = 0
        while (correct_classification/test_data_size) <= exit_accuracy and iter < max_iterations:
            iter += 1
            correct_classification = 0
            classifier.train(train_data, train_labels, learning_rate, regularization_factor)

            for test_index in range(len(test_data)):

                if test_labels[test_index] == classifier.predict(test_data[test_index]):
                    correct_classification += 1
                #else:
                    #print(iter)
                    #print(predicted)
                    #print('E:' + str(test_labels[test_index]))
        classifiers.append(classifier)
        accuracies.append(correct_classification/test_data_size)

    print(accuracies)
    """classifier = LogisticClassifier(number_of_features, label_values)
    correct_classification = 0
    while (correct_classification / all_data_size) >= exit_accuracy and iter < max_iterations:
        iter += 1
        correct_classification = 0
        classifier.train(data, labels, learning_rate, regularization_factor)
        for test_index in range(len(data)):
            if labels[test_index] == classifier.predict(data[test_index]):
                correct_classification += 1"""

    correct_classification = 0
    for test_index in range(len(data)):
        predictions = []
        for classifier in classifiers:
            predictions.append(classifier.predict(data[test_index]))
        prediction_labels = set(predictions)
        label_counts = {}
        for label in prediction_labels:
            label_counts[label] = 0
        for prediction in predictions:
            label_counts[prediction] += 1
        predicted = max(label_counts, key=label_counts.get)
        if labels[test_index] == predicted:
            correct_classification += 1
        else:
            #print(iter)
            print('E:' + str(labels[test_index]) + 'P:' + predicted)
            print(label_counts)

    print(str((correct_classification / all_data_size)))


    return classifiers

