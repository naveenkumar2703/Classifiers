"""
class - Node acts a data holder for decision tree
@author: Naveen
"""

operators = ['E', 'LE', 'GE', 'NE', 'L', 'G', 'IN']
class Node:

    def __init__(self, parent):
        self.parent = parent
        if parent is None:
            self.level = 1
        else:
            self. level = parent.level + 1
        self.root_node = False
        self.leaf_node = False
        self.label = None
        self.attribute = None
        self.conditional_operator = None
        self.criteria_value = None
        self.children = None

    def set_test_condition(self, attribute, conditional_operator, criteria_value):
        self.attribute = attribute
        if conditional_operator not in operators:
            raise ValueError('operator should be one of '+str(operators))
        self.conditional_operator = conditional_operator
        if criteria_value is None:
            raise ValueError('value for test is missing.')
        if type(criteria_value) is list and len(criteria_value) == 0:
            raise ValueError('value is empty list.')
        if (not type(criteria_value) is list) and  conditional_operator is 'IN':
            raise ValueError('IN is applicable only for range of values')

        self.criteria_value = criteria_value[:] # creatung a copy of values
        #TODO create child nodes for each condition
        children = {}
        for value in criteria_value:
            children[value] = Node(self)
        self.children = children


    def evaluate(self, model, data_point):
        child_node = None
        if type(self.criteria_value) is list:
            if self.conditional_operator is 'E':
                for item in self.criteria_value:
                    if model.get_attribute_value(self.attribute, data_point) == item:
                        child_node = self.children[item]

            elif self.conditional_operator is 'IN':
                for item in self.criteria_value:
                    if model.get_attribute_value(self.attribute, data_point) < item: #assuming numeric data is in order
                        child_node = self.children[item]

        else:
            if self.conditional_operator is 'E':
                if model.get_attribute_value(self.attribute, data_point) is self.criteria_value:
                    child_node = self.children['T']
                else:
                    child_node = self.children['F']

            elif self.conditional_operator is 'L':
                if model.get_attribute_value(self.attribute, data_point) < self.criteria_value:
                    child_node = self.children['T']
                else:
                    child_node = self.children['F']

            elif self.conditional_operator is 'G':
                if model.get_attribute_value(self.attribute, data_point) > self.criteria_value:
                    child_node = self.children['T']
                else:
                    child_node = self.children['F']

            elif self.conditional_operator is 'LE':
                if model.get_attribute_value(self.attribute, data_point) <= self.criteria_value:
                    child_node = self.children['T']
                else:
                    child_node = self.children['F']

            elif self.conditional_operator is 'GE':
                if model.get_attribute_value(self.attribute, data_point) >= self.criteria_value:
                    child_node = self.children['T']
                else:
                    child_node = self.children['F']

            elif self.conditional_operator is 'NE':
                if not model.get_attribute_value(self.attribute, data_point) is self.criteria_value:
                    child_node = self.children['T']
                else:
                    child_node = self.children['F']

        return child_node

    def __str__(self):
        if self.leaf_node or self.children is None:
            return 'Leaf node has label - '+ str(self.label) +'. '
        else:
            return 'Node has attribute - ' + self.attribute\
               + ' operator - ' + str(self.conditional_operator)\
               + ' and values -' + str(self.criteria_value)\
               + 'with label '+ str(self.label)+ ' at level '+ str(self.level)+' . '








