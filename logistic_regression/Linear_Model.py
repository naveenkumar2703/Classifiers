
import numpy as np

class Linear_Model:

    def __init__(self, number_of_features, number_of_labels):
        self.number_of_features = number_of_features
        self.number_of_labels = number_of_labels
        self.weights = np.zeros(number_of_labels, number_of_features)

