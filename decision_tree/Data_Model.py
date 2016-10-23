"""
@author: Naveen

This class has the decision tree, attributes and its possible values
"""

class Data_Model:
    def __init__(self):
        self.attributes = None
        self.attribute_type = None
        self.attribute_data_index = None
        self.attribute_data_values = None
        self.root_node = None
        self.exit_depth = 99999

    """
    This method constructs a model based on given training example. 'attribute_data_values' might change if new example set is given with new value.
    Input:
    data - n * m matrix with n - data points and m - attribute_values
    header - array of size m with attribute names
    """
    def construct_model(self, data, header):
        if len(data[0]) != len(header):
            raise ValueError('Number of attributes in header and data is not matching')

        attribute_data_index = {}
        attribute_data_values = {}
        for index in range(len(header)):
            if header[index] in attribute_data_index:
                raise ValueError('Duplicate attributes are present')
            attribute_data_index[header[index]] = index
            attribute_values = []
            for point in data:
                if point[index] not in attribute_values:
                    attribute_values.append(point[index])
            attribute_data_values[header[index]] = attribute_values

            if len(attribute_values) == len(data):
                print('WARNING: Attribute: ' + str(header[index])+' looks like identifier. Remove it before training')
            elif len(attribute_values) > 24:
                print('WARNING: Attribute: ' + str(header[index]) + ' looks like continuous value. Discretize it before training.')
            elif len(data)/len(attribute_values) < 2:
                print('WARNING: Attribute: ' + str(header[index]) + ' has many different values.')

        self.data = data
        self.attributes = header
        self.attribute_data_index = attribute_data_index
        self.attribute_data_values = attribute_data_values

    def get_attribute_value(self, attribute, data_point):
        return data_point[self.attribute_data_index[attribute]]

    def predict(self, data_point):
        node = self.root_node.evaluate(self,data_point)

        prev_label = self.root_node.label
        while node is not None and not node.leaf_node:
            prev_label = node.label
            if node.level > self.exit_depth:
                node = None # no need to evaluate if exit depth is met
            else:
                node = node.evaluate(self,data_point)

        label = None
        if node is None:
            label = prev_label # node is None means there is no such branch from training. So returning a value from a parent majority
        else:
            label = node.label

        return label

    def __str__(self):
        tree = []
        #tree[0].append(self.root_node)
        curr_level_nodes = [self.root_node]
        while len(curr_level_nodes) > 0:
            tree.append([])
            next_level = []
            for node in curr_level_nodes:
                tree[len(tree) - 1].append(node)
                if not node.leaf_node and node.children is not None:
                    next_level.extend(list(node.children.values()))
            curr_level_nodes = next_level

        max_length_of_tree = 1
        for level in tree:
            if len(level) > max_length_of_tree:
                max_length_of_tree = len(level)

        depth = 0
        for level in tree:
            depth += 1
            print('Level - ' + str(depth))
            for node in level:
                print(node,end="")
            print("")

        return "Printed model with depth - " \
               + str(depth) + ' and maximum width ' \
               + str(max_length_of_tree)




