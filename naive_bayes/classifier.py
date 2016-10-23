#############################
#Author: Naveenkumar
# File: naive_bayes - classifier
#############################

import numpy as np
from math import exp
from math import sqrt
from math import pi

"""
Inputs:
    - data as Dictionary. Feature_name as key with value
    - label_probabilities - Dictionary - label:probability for each labels.
    - feature_label_probabilities - Dictionary - feature: Dictionary - label: probability given label - list of mran 2* var and sqrt for numeric

Output:
    - Discrete label with highest probability
"""



def find_most_probable_label(data, label_priors, feature_label_probabilities):
    label_probabilities = {}
    for label in label_priors:
        current_probability = 1 # initializing

        for feature in feature_label_probabilities:
            if type(feature_label_probabilities[feature][label]) == type({}):
                current_probability *= feature_label_probabilities[feature][label][data[feature]]
            else:
                mean = feature_label_probabilities[feature][label][0]
                twice_var = feature_label_probabilities[feature][label][1]
                sqrt_2pi_var = feature_label_probabilities[feature][label][2]
                estimated_numeric_probability = 1
                if twice_var == 0 or sqrt_2pi_var == 0:
                    estimated_numeric_probability = 1
                else:
                    #estimating numeric probability using gaussian distribution from mean and variance
                    estimated_numeric_probability = (exp(-(((float(data[feature]) - mean) ** 2)/twice_var)))/sqrt_2pi_var
                current_probability *= estimated_numeric_probability
        label_probabilities[label] = current_probability * label_priors[label]
    return max(label_probabilities, key=label_probabilities.get)

# labels of all training examples
def compute_label_priors(labels):
    label_values = set(labels)
    label_counts = {}
    #initializing
    for item in label_values:
        label_counts[item] = 0

    for item in labels: # counting
        label_counts[item] += 1

    return  compute_label_prior_from_count(label_counts)


def compute_label_prior_from_count(label_counts):
    label_priors = {}
    count = 0
    for label in label_counts:
        count += label_counts[label]

    for label in label_counts: # turning counts to probabilities
        label_priors[label] = label_counts[label]/count

    return label_priors

"""
Input -
    - features: list of features which is input to the model
    - labels: list of list of label in the same order as features

Output:
    - Dictionary with possible values of feature as key with vale as dictionary that takes one of labels as key and prior as value.
"""
def compute_discrete_feature_probability(feature_values, labels):
    label_items = set(labels)
    feature_item_values = set(feature_values)

    feature_val_label_counts = {}
    for value in feature_item_values: #initializing
        feature_val_label_counts[value] = {}
        for label in label_items:
            feature_val_label_counts[value][label] = 0

    label_counts = {}
    feature_probabilities = {}
    # initializing
    for item in label_items:
        label_counts[item] = 0
        feature_probabilities[item] = {}

    for index in range(len(feature_values)): # counting
        feature_val_label_counts[feature_values[index]][labels[index]] += 1
        label_counts[labels[index]] += 1


    for value in feature_item_values: #computing probabilities
        for label in label_items:
            feature_probabilities[label][value] = (feature_val_label_counts[value][label] + 1)/(label_counts[label] + len(label_items))
            # +1 and + len(label_items) is for smoothing

    return feature_probabilities

def compute_numeric_feature_probability(feature_values, labels):
    label_items = set(labels)
    data_by_label = {}
    feature_probabilities = {}
    for label in label_items: # initializing
        data_by_label[label] = []
        feature_probabilities[label] = []

    for index in range(len(feature_values)):  # separating data by label
        data_by_label[labels[index]].append(feature_values[index])

    for label in data_by_label:
        curr_data = np.array(data_by_label[label]).astype(float)
        feature_probabilities[label].append(np.mean(curr_data)) # appending mean at zeroth index
        variance = np.var(curr_data)
        feature_probabilities[label].append((2 * variance)) # adding twice var at index - 1
        feature_probabilities[label].append(sqrt(2 * pi * variance)) # appending sqrt 2 * pi * var at index - 2

    return feature_probabilities



def compute_featues_probabilities(data, labels, feature_name_types):
    feature_probabilities = {}
    #TODO call discrete & define continuous.
    for feature in feature_name_types:
        feature_type = feature_name_types[feature]
        feature_values = []
        for item in data:
            if item.get(feature, None) is not None:
                feature_values.append(item.get(feature))

        if feature_type == 'DISCRETE':
            feature_probabilities[feature] = compute_discrete_feature_probability(feature_values, labels)
        else:
            feature_probabilities[feature] = compute_numeric_feature_probability(feature_values, labels)
    return feature_probabilities