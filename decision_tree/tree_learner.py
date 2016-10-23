"""

"""
from impurity.measure import calculate_entropy
from decision_tree.Node import Node
from collections import Counter


def grow_tree(model, parent_node, branch_value, attributes, data, labels):
    entropy = calculate_entropy(labels)
    node = None
    if parent_node is None: # initialize root node
        node = Node(parent_node)
        model.root_node = node
        node.root_node = True
    else:
        node = parent_node.children[branch_value]

    # Once node is obtained set majority class label to handle unseen test examples.
    # finding label with maximum occurrence
    max_label = max(set(labels), key=labels.count)
    node.label = max_label  # setting majority class as label

    if entropy == 0.0:
        node.leaf_node = True # Setting as leaf node as nothing more to split

    else:
        if len(attributes) == 1: # if this is the final attribute
            node.leaf_node = True

        else:
            gain_for_attributes = []
            attribute_data_label = {}
            for index in range(len(attributes)):
                attribute = attributes[index]
                att_data_index = model.attribute_data_index[attribute]
                att_values = model.attribute_data_values[attribute]
                sum_gain_for_attribute = 0
                att_value_counts = {}
                for value in att_values:
                    att_value_counts[value] = [[],[]] # initializing for data and labels
                for data_index in range(len(data)):
                    data_point = data[data_index]
                    label_point = labels[data_index]
                    new_data_list = att_value_counts[data[data_index][att_data_index]][0]
                    new_data_list.append(data_point)
                    new_label_list = att_value_counts[data[data_index][att_data_index]][1]
                    new_label_list.append(label_point)

                for value in att_values:
                    sum_gain_for_attribute += ((len(att_value_counts[value][1])/len(labels)) * (calculate_entropy(att_value_counts[value][1])))
                gain_for_attributes.append(entropy - sum_gain_for_attribute)
                attribute_data_label[attribute] = att_value_counts
            att_index_with_max_gain = gain_for_attributes.index(max(gain_for_attributes))
            attribute_values = model.attribute_data_values[attributes[att_index_with_max_gain]]
            node.set_test_condition(attributes[att_index_with_max_gain], 'E', attribute_values)
            attribute = attributes[att_index_with_max_gain]
            attributes.remove(attribute)

            if len(attributes) > 0:
                for value in attribute_values:
                    if len(attribute_data_label[attribute][value][1]) > 0:
                        grow_tree(model, node, value, attributes, attribute_data_label[attribute][value][0],
                                  attribute_data_label[attribute][value][1])
            attributes.insert(att_index_with_max_gain, attribute)

            for value in attribute_values:
                if node.children[value].children is None and node.children[value].leaf_node is False:
                    node.children.pop(value)
                    node.criteria_value.remove(value)


def train(model, data, labels):
    total_entropy = calculate_entropy(labels)


    if total_entropy == 0:
        root_node = Node(None)
        model.root_node = root_node
        root_node.root_node = True
        root_node.leaf_node = True
        root_node.label = labels[0]  # all labels are same

    else:
        grow_tree(model, None,None, model.attributes, data, labels)

    return model
