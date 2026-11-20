# Created by tnrbf@umsystem.edu on 2026-11-13

# Last updated by tnrbf@umsystem.edu on 2026-11-13

import math

from graphviz import Digraph

class Decision_Tree:
    # Constructor
    def __init__(self):
        """ Constructs a decision tree classifier.
            :rtype: None
        """
        # Data fields
        self._root = None
    # Getters
    def get_root(self):
        """ Returns the root node of the decision tree.
            :returns: root of the decision tree
            :rtype: dict    
        """
        return self._root
    # Methods
    def get_class_labels(self, data):
        """ Extracts the class labels from the dataset.
            :param data: training data
            :type data: list
            :returns: class labels of the training data
            :rtype: list
        """
        return [record["Class"] for record in data]
    def counts_occurrences(self, data, attribute):
        """ Counts the occurrences of each value in an attribute.
            :param data: training data
            :type data: list
            :param attribute: attribute to count occurrences for
            :type attribute: str
            :returns: a dictionary with the counts of each value in the attribute
            :rtype: dict
        """
        counts = {}
        for record in data:
            value = record[attribute]
            if value not in counts:
                counts[value] = 0
            counts[value] += 1
        return counts
    def calculate_entropy(self, data, attribute="Class"):
        """ Calculates the entropy of a set of class labels.
            :param data: training data
            :type data: list
            :param attribute: attribute to calculate entropy for
            :type attribute: str
            :returns: the calculated entropy
            :rtype: float
        """
        if (len(data) == 0):
            return 0.0

        # Counts the occurrences of each class
        counts = self.counts_occurrences(data, attribute)

        # Calculates the entropy
        entropy = 0.0
        total = len(data)

        for count in counts.values():
            if count > 0:
                probability = count / total
                entropy -= probability * math.log2(probability) # entropy = -sum(p * log2(p))
        return entropy
    def group_by_attribute(self, data, attribute):
        """ Groups the data by the values of an attribute.
            :param data: training data
            :type data: list
            :param attribute: attribute to group by
            :type attribute: str
            :returns: a dictionary with the grouped data
            :rtype: dict
        """
        groups = {}
        for record in data:
            value = record[attribute]
            if value not in groups:
                groups[value] = []
            groups[value].append(record)
        return groups
    def calculate_weighted_entropy(self, groups):
        """ Calculates the weighted entropy of groups records.
            :param groups: a dictionary with the grouped data
            :type groups: dict
            :returns: the calculated weighted entropy
            :rtype: float
        """
        total_records = sum(len(group) for group in groups.values())
        if total_records == 0:
            return 0.0
        
        weighted_entropy = 0.0

        for group in groups.values():
            probability = len(group) / total_records
            entropy = self.calculate_entropy(group, "Class")
            weighted_entropy += probability * entropy # weighted_entropy = sum(p * entropy for each group)

        return weighted_entropy
    def calculate_information_gain(self, data, attribute):
        """ Calculates the information gain of an attribute.
            :param data: training data
            :type data: list
            :param attribute: attribute used to split the data
            :type attribute: str
            :returns: the calculated information gain
            :rtype: float
        """
        # Calculate the entropy of the original dataset
        original_entropy = self.calculate_entropy(data, "Class")

        # Group the data by the specified attribute
        groups = self.group_by_attribute(data, attribute)

        # Calculate the weighted entropy of the groups
        weighted_entropy = self.calculate_weighted_entropy(groups)

        # Calculate the information gain
        information_gain = original_entropy - weighted_entropy # information_gain = entropy_before_split - weighted_entropy_after_split

        return information_gain
    def find_best_attribute(self, data, attributes):
        """ Finds the attribute with the highest information gain.
            :param data: training data
            :type data: list
            :param attributes: list of attributes to consider
            :type attributes: list
            :returns: the attribute with the highest information gain
            :rtype: str
        """
        best_gain = float('-inf')  # Initialize best_gain to negative infinity
        best_attribute = None

        for attribute in attributes:
            gain = self.calculate_information_gain(data, attribute)
            if gain > best_gain:
                best_gain = gain
                best_attribute = attribute
        return best_attribute
    def find_majority_class(self, data):
        """ Finds the majority class in the dataset.
            :param data: training data
            :type data: list
            :returns: the majority class label
            :rtype: str
        """
        # Counts the occurrences of each class
        counts = self.counts_occurrences(data, "Class")

        # Finds the class with the highest count
        majority_class = max(counts, key=counts.get)

        return majority_class
    def is_pure(self, data):
        """ Checks if the dataset is pure (all instances belong to the same class).
            :param data: training data
            :type data: list
            :returns: True if the dataset is pure, False otherwise
            :rtype: bool
        """
        if len(data) == 0:
            return True  # An empty dataset can be considered pure

        counts = self.counts_occurrences(data, "Class")
        return len(counts) == 1  # Pure if there's only one unique class
    def build_tree(self, data, attributes):
        """ Recursively builds the decision tree.
            :param data: training data
            :type data: list
            :param attributes: list of attributes to consider for splitting
            :type attributes: list
            :returns: the root node of the decision tree
            :rtype: dict
        """
        if not data:
            raise ValueError("Data cannot be empty when building the tree.")

        # Base case: If the dataset is pure, return a leaf node with the class label
        if self.is_pure(data):
            return {"class": data[0]["Class"]}

        # Base case: If there are no more attributes to split on, return a leaf node with the majority class
        if not attributes:
            majority_class = self.find_majority_class(data)
            return {"class": majority_class}

        # Find the best attribute to split on
        best_attribute = self.find_best_attribute(data, attributes)

        information_gain = self.calculate_information_gain(data, best_attribute)

        if information_gain <= 0:
            return {"class": self.find_majority_class(data)}

        # Create the current decision node
        node = {"attribute": best_attribute, "majority_class": self.find_majority_class(data), "branches": {}}

        # Group records according to the selected attribute
        groups = self.group_by_attribute(data, best_attribute)

        # Removes the selected attribute from future splits
        remaining_attributes = [attr for attr in attributes if attr != best_attribute]

        # Recursively builds each subtree
        for value, group in groups.items():
            node["branches"][value] = self.build_tree(group, remaining_attributes)

        return node
    def train(self, data, attributes):
        """ Trains the decision tree classifier.
            :param data: training data
            :type data: list
            :param attributes: list of attributes to consider for splitting
            :type attributes: list
            :returns: None
        """
        if len(data) == 0:
            raise ValueError("Training data cannot be empty.")
        
        self._root = self.build_tree(data, attributes)
    def draw_tree(self, filename="decision_tree"):
        """ Draws the decision tree.
            :param filename: name of the output file (without extension)
            :type filename: str
            :returns: None
        """
        if self._root is None:
            raise ValueError("The decision tree has not been trained yet.")

        # Create a new directed graph
        graph = Digraph(format='svg', engine='dot')

        # Sets the graph layout
        graph.attr(rankdir='TB', nodesep='0.6', ranksep='0.8')  # Top to Bottom layout

        # Sets the default node appearance
        graph.attr('node', 
                   fontname="Arial", fontsize="16", fixedsize="true", 
                   width="0.65", height="0.65", margin="0", 
                   style="filled", fillcolor="white", color="black")

        # Sets the default edge appearance
        graph.attr('edge', fontname="Arial", fontsize="12", fontcolor="blue", color="black")

        # Draws the tree
        self._draw_node(graph, self._root, "0")

        # Saves the graph
        graph.render(filename, cleanup=True)
    def _draw_node(self, graph, node, node_id):
        """ Recursively draws a node and its children in the graph.
            :param graph: the graph object
            :type graph: Digraph
            :param node: the current node to draw
            :type node: dict
            :param node_id: unique identifier for the current node
            :type node_id: str
            :returns: None
        """
        # Case 1: Leaf node
        if "class" in node:
            graph.node(node_id, label=node["class"], shape="ellipse")
            return
        # Case 2: Decision node
        graph.node(node_id, label=node["attribute"], shape="box")

        # Draws branches
        for value, child in node["branches"].items():
            child_id = f"{node_id}_{value}"
            self._draw_node(graph, child, child_id)
            graph.edge(node_id, child_id, label=str(value))
    def _predict_node(self, node, instance):
        """ Predicts the class of an instance
            :param node: the current node
            :type node: dict
            :param instance: the instance to classify
            :type instance: dict
            :returns: the predicted class label
            :rtype: str
        """
        # Base case: If the node is a leaf, return the class label
        if "class" in node:
            return node["class"]

        # Get the attribute to split on
        attribute = node["attribute"]
        # Get the value of the attribute in the instance
        value = instance[attribute]
        # Check whether the branch exists
        if value not in node["branches"]:
            # If the branch does not exist, return the majority class of the current node
            return node["majority_class"]
        # Recursively predicts
        child = node["branches"][value]
        return self._predict_node(child, instance)
    def predict(self, instance):
        """ Predicts the class of an instance.
            :param instance: the instance to classify
            :type instance: dict
            :returns: the predicted class label
            :rtype: str
        """
        if self._root is None:
            raise ValueError("The decision tree has not been trained yet.")
        return self._predict_node(self._root, instance)