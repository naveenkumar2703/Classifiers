########################################################################################################################
# Author: Naveenkumar Ramaraju
# Class: binary_tree_classifier
# This builds a binary tree classifier for given attribute, data and labels.
#
########################################################################################################################
# imports
from math import log2
from Node import *

# class for binay classifier
class binary_tree_classifier(object):

    # constructor
    def __init__(self):
        self.root_node = None
        self.attributes = None  # holds features of the classifier
        self.attribute_type = None
        self.attribute_values = None   # holds possible values for each feature in a dict
        self.maximum_depth = 99999  # Initializing with arbitrarily large value

    # This method trains the classifier for given model, data and labels
    def train(self, attributes, data, labels):

        if len(attributes) != len(set(attributes)): # validating for duplicates
            raise Exception('Attribute names contain duplicates.')

        if len(data[0]) != len(attributes):  # Sample check on data size and feature size match
            raise Exception("Dimensions of attribute and data doesn't match")

        if self.attribute_values is None:  # If attribute values are not provided before training build it from data
            self.construct_attribute_values(attributes, data)

        self.attributes = attributes # setting attributes to the classifer
        total_entropy = calculate_entropy(labels) # calculate entropy

        if total_entropy == 0: # If all labels are same. Stop.
            root_node = Node(None)
            self.root_node = root_node
            root_node.root_node = True
            root_node.leaf_node = True
            root_node.label = labels[0]  # all labels are same

        else:
            self.grow_tree(None, data, labels) # Grow tree.

        return self # Return the tree

    # This method grows tree recursively till all data reaches same label or maximum depth is reached
    def grow_tree(self, parent_node, data, labels):
        node = Node(parent_node)  # Creating a new node
        if parent_node is None: # initialize root node
            node.is_root = True
            self.root_node = node

        # Checking if tree has reached maximum depth limit
        if (parent_node is not None and parent_node.depth >= self.maximum_depth) \
                or (labels is not None and len(labels) == 0):
            return None  # Stop growing tree if true.

        # finding label with maximum occurrence
        max_label = max(set(labels), key=labels.count)
        node.label = max_label  # setting majority class as label to handle unseen test examples.

        # Checking entropy on labels
        entropy = calculate_entropy(labels)
        node.entropy = entropy # set entropy of data at the node.

        if entropy == 0: # All labels are same.
            node.is_leaf = True # make this as leaf node

        else: # Proceed
            attributes_value_data_labels = {}  # This will hold counts of  label for all attribute label value pair
            attributes_values_info_gain = {}  # This will hold information gain for each attribute value pairs
            att_info_gain = {} # this will be used for non numeric to find best attribute
            for attribute in self.attributes:
                attribute_value_labels = {}  # This will hold label count value for all possible values of an attribute

                for value_index in range(len(self.attribute_values[attribute])):
                    key = self.attribute_values[attribute][value_index]
                    if self.attribute_type[attribute] is 'num': # if numeric data
                        key = float(key) # convert it to float for comparision using < & >
                    attribute_value_labels[key] = [[],[]]  # Initialize 2-d array to hold data and label in pairs in same sequence

                attributes_value_data_labels[attribute] = attribute_value_labels # put it in dict for each attribute
                attributes_values_info_gain[attribute] = {} # Initialize dict for each attribute

            for data_index in range(len(data)): # for each data point
                data_point = data[data_index]
                curr_label = labels[data_index]
                for attribute_index in range(len(self.attributes)): # for each attribute
                    curr_att_dict = self.attributes[attribute_index]
                    curr_type = self.attribute_type[curr_att_dict]
                    curr_value = data_point[attribute_index]
                    if curr_type is 'num': # convert number to float
                        curr_value = float(curr_value)
                    att_data_label_list = attributes_value_data_labels[curr_att_dict][curr_value] # accessing data and label list
                    att_data_label_list[0].append(data_point)  # Adding data point to its list
                    att_data_label_list[1].append(curr_label)  # Adding label for the data

            # Compute information gain based on count of segregated data for all attribute values
            for attribute in self.attributes:
                att_type = self.attribute_type[attribute]
                if att_type is 'num': # if continuous
                    values = self.attribute_values[attribute][:] # get all possible values in the data. This is not efficient in big data. Need to form bins.
                    values.sort() # sorting the values to form bins. Inefficient over large range
                    labels_inc_split = [] # List to hold the labels as the value increases, to form cumulative bin
                    for value in values:
                        labels_inc_split.extend(attributes_value_data_labels[attribute][value][1])# Adding values for next split
                        if len(labels_inc_split) > 0: # if a bin has some data
                            ratio = len(labels_inc_split)/len(labels)
                            ent = calculate_entropy(labels_inc_split) # find entropy
                            attributes_values_info_gain[attribute][value] = (entropy - (ratio * ent)) # store gain for attribute value pair

                else:
                    for attribute in self.attributes:
                        values = self.attribute_values[attribute][:] # getting all possible values for the attribute
                        sum_gain_for_attribute = 0
                        for value in values:
                            labels_for_att_val = attributes_value_data_labels[attribute][value][1] # No need for cumulative addition
                            if len(labels_for_att_val) > 0: # check if an attribute value has a label associated with it
                                ratio = len(labels_for_att_val)/len(labels) # find its proportion
                                ent = calculate_entropy(labels_for_att_val) # get entropy
                                ratio_by_ent = (ratio * ent) # calculate attribute value's share of entropy
                                sum_gain_for_attribute += ratio_by_ent # sum it up for attribute
                                attributes_values_info_gain[attribute][value] = ratio_by_ent # store for attribute value
                        att_info_gain[attribute] = entropy - sum_gain_for_attribute # store for attribute


            # Declaring variables to pick next feature
            max_num_gain = -99990 # To hold continuous numeriv value gain. Initializing with arbitrarily large negative num
            max_gain_attr = None
            max_gain_val = None
            max_disc_gain = -9999 # To hold discrete values gain. Initializing with arbitrarily large negative num
            max_disc_att = None
            # Going through all the attributes to pick the best
            for attribute in self.attributes:
                att_type = self.attribute_type[attribute]
                if att_type is not None and att_type is 'num': # handling numeric
                    for val_info_gain in attributes_values_info_gain[attribute]: # for each information gain
                        curr_gain = attributes_values_info_gain[attribute][val_info_gain]
                        if max_num_gain <= curr_gain: # replace if current one is best
                            max_num_gain = curr_gain
                            max_gain_attr = attribute
                            max_gain_val = val_info_gain
                else:
                    min_loss = 9999 # Initializing arbitrarily large number
                    if att_info_gain[attribute] > max_disc_gain: # checking for the best attribute gain
                        max_disc_gain = att_info_gain[attribute]
                        max_disc_att = attribute

                        values = self.attribute_values[attribute][:] # getting all the values of the attribute.
                        for value in values:
                            info_gain = attributes_values_info_gain[attribute].get(value,None) # passing None to return None if key not found
                            if info_gain is not None and min_loss > info_gain: # checking for best value that can be tested in next node
                                min_loss = attributes_values_info_gain[attribute][value]
                                max_gain_val = value

            if max_disc_gain > max_num_gain: # in case of mix of numeric and non numeric attributes, picking the best
                max_gain_attr = max_disc_att


            if max_gain_attr is None: # If no attribute has gain set it as leaf node
                node.is_leaf = True
                node.entropy = len(labels)
            else: # Else create child nodes
                # Setting the values to node
                node.attribute = max_gain_attr
                node.attribute_value = max_gain_val
                node.attribute_index = self.attributes.index(max_gain_attr)
                if max_disc_gain < max_num_gain: # setting node attribute type to do correct testing while prediction.
                    node.attribute_type = 'num'

                # Accessing data to build child nodes
                true_data = []# holder for data if condition holds
                true_labels = []# holder for label of data for which condition holds
                false_data = [] # for flase scenario
                false_labels = [] # for flase scenario
                if max_disc_gain > max_num_gain: #handling non numeric attribute
                    for value in self.attribute_values[max_gain_attr]:
                        curr_data = attributes_value_data_labels[max_gain_attr][value][0]
                        curr_label = attributes_value_data_labels[max_gain_attr][value][1]

                        if value is max_gain_val and \
                                        curr_data is not None and len(curr_data) > 0 and \
                                        curr_label is not None and len(curr_label) > 0: # Only difference from numeric is equality check instead of '<'

                            true_data.extend(curr_data)
                            true_labels.extend(curr_label)
                        elif curr_data is not None and len(curr_data) > 0 and \
                                        curr_label is not None and len(curr_label) > 0: # False scenario
                            false_data.extend(curr_data)
                            false_labels.extend(curr_label)

                else: # handling continuous values
                    for value in self.attribute_values[max_gain_attr]:
                        curr_data = attributes_value_data_labels[max_gain_attr][value][0]
                        curr_label = attributes_value_data_labels[max_gain_attr][value][1]
                        if value <= max_gain_val and\
                                curr_data is not None and len(curr_data ) > 0 and\
                                curr_label is not None and len(curr_label) > 0: # forming two bins of data based on '<=' condition

                            true_data.extend(curr_data)
                            true_labels.extend(curr_label)
                        elif curr_data is not None and len(curr_data ) > 0 and\
                                curr_label is not None and len(curr_label) > 0: # If condition doesn't hold.
                            false_data.extend(curr_data)
                            false_labels.extend(curr_label)

                # Growing child nodes
                node.true_child = self.grow_tree(node, true_data, true_labels)
                node.false_child = self.grow_tree(node, false_data, false_labels)

                if node.true_child is None and node.false_child is None: # If unable to construct child node
                    node.is_leaf = True # Make this node as leaf.
                    node.entropy = len(labels) # Know the length of label as entropy if needed in future.

        return node # returning the constructed node.




    # This method finds out possible values each attribute can take from the training set.
    # For very large data set it is preferable to provide this.
    def construct_attribute_values(self, attributes, data):
        attribute_values = {}
        self.attribute_type = {}
        for attribute in attributes:
            attribute_values[attribute] = []  # create empty list to hold list of values for each attribute

        for data_point in data: # Look for values in each data for each attribute
            for index in range(len(attributes)):
                curr_value = data_point[index]
                curr_attribute = attributes[index]
                if curr_value not in attribute_values[curr_attribute]:  # add it if not present
                    attribute_values[curr_attribute].append(curr_value)

        for curr_attribute in attributes:
            temp = attribute_values[curr_attribute]
            try:
                # TODO - optional
                # comment below two lines, if numbers needed to be handled as discrete values instead of continuous
                attribute_values[curr_attribute] = [float(num) for num in attribute_values[curr_attribute]]
                self.attribute_type[curr_attribute] = 'num'
            except: # If error occurs, this attribute is not a number.
                self.attribute_type[curr_attribute] = 'str'
                attribute_values[curr_attribute] = temp

        self.attribute_values = attribute_values  # Assign constructed values to self

    # method to predict label for given data
    def predict(self, data):
        for att_index in range(len(self.attributes)):
            if self.attribute_type[self.attributes[att_index]] is 'num':
                data[att_index] = float(data[att_index]) # converting string input to number if type is 'num'

        return self.root_node.evaluate(data) # predicting by calling evaluate on root_node

    # This method prints the tree upon call of print function.
    def __str__(self):
        tree = []
        curr_level_nodes = [self.root_node]
        while len(curr_level_nodes) > 0:
            tree.append([])
            next_level = []
            for node in curr_level_nodes:
                tree[len(tree) - 1].append(node)
                if not node.is_leaf:
                    if node.true_child is not None:
                        next_level.append(node.true_child)

                    if node.false_child is not None:
                        next_level.append(node.false_child)
            curr_level_nodes = next_level

        max_length_of_tree = 1
        for level in tree:
            if len(level) > max_length_of_tree:
                max_length_of_tree = len(level)

        depth = -1
        delim = '-->'
        for level in tree:
            depth += 1
            print(delim * (depth),end="")
            for node in level:
                print(node,end="")
            print("")

        return "Printed model with depth - " \
               + str(depth) + ' and maximum width ' \
               + str(max_length_of_tree)


# This function calculates entropy for given labels
def calculate_entropy(labels):
    label_values = set(labels) # Getting unique values from labels
    label_counts = {}
    for item in label_values:
        label_counts[item] = 0 # initializing the tuple for each item in the label
    for item in labels: # count the label
        label_counts[item] += 1

    entropy = 0

    for item in label_values: # calculate entropy for each value in the label
        pi = label_counts[item]/len(labels)
        entropy += (-(pi * log2(pi)))
    return entropy






