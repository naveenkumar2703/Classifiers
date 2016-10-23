########################################################################################################################
# Author: Naveenkumar Ramaraju
# Class: Node
# This class is a data structure to hold attributes of node in a binary tree
########################################################################################################################
class Node(object):


    # Constructor for Node class
    def __init__(self, parent):
        self.parent = parent
        self.is_root = False  # Place holder for variable
        self.is_leaf = False  # Place holder for variable
        self.true_child = None  # Place holder for variable to hold child if node tests to true
        self.false_child = None  # Place holder for variable to hold child if node tests to false
        self.attribute = None  # Place holder for variable that holds attribute that will be tested at this node
        self.attribute_index = None  # Place holder to have index of the attribute in the feature vector
        self.attribute_value = None  # Place holder for variable that holds the attribute value to be tested against
        self.label = None  # Place holder for variable that holds class label for majority class at this node
        self.depth = None # Place holder for variable that holds the depth at which this node is located
        self.attribute_type = None
        if parent is None:
            self.depth = 0
        else:
            self.depth = parent.depth + 1
        self.entropy = None


    # This method evaluates the node and calls its children or returns class label
    def evaluate(self, data):
        if self.is_leaf: # Checking if this is leaf node.
            return self.label  # return the class label if it is leaf

        elif data is not None and type(data) is list\
            and len(data) > 0 and data[self.attribute_index] is not None: # checking if data has value for attribute at its index
            if self.attribute_type is None or self.attribute_type is not 'num':
                if data[self.attribute_index] == self.attribute_value and self.true_child is not None: # Equality check for non numbers
                    return self.true_child.evaluate(data)
                elif self.false_child is not None:
                    return self.false_child.evaluate(data)
            elif self.attribute_type is 'num':
                if data[self.attribute_index] <= self.attribute_value and self.true_child is not None:
                    return self.true_child.evaluate(data)
                elif self.false_child is not None:
                    return self.false_child.evaluate(data)

        # If not able to predict further return majority class
        return self.label


    def __str__(self):
        if self.is_leaf or (self.true_child is None and self.false_child is None):
            return 'Class label - ' + str(self.label) + '. '
        else:
            return  self.attribute + str()\
                   + ' is - ' + str(self.attribute_value)+'. ' \
                   #+ 'with label ' + str(self.label) + ' at level ' + str(self.depth) + ' entropy'+ str(self.entropy)+' . '
