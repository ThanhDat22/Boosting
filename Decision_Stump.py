# Created by tnrbf@umsystem.edu on 2026-11-13

# Last updated by tnrbf@umsystem.edu on 2026-11-19

from Decision_Tree import Decision_Tree

class Decision_Stump(Decision_Tree):
    # Constructor
    def __init__(self):
        """ Constructs a one-level decision tree classier.
        :rtype: None 
        """
        super().__init__()
    # Methods
    def split_data(self, data, attribute, threshold):
        """ Splits numerical data into two groups using a threshold.
            :param data: The data to split.
            :type data: list
            :param attribute: numerical attribute used for splitting.
            :type attribute: str
            :param threshold: value used to split the records
            :type threshold: float
            :returns: groups for values <= and > threshold
            :rtype: dict
        """
        groups = {True: [], False: []}

        for record in data:
            if record[attribute] <= threshold:
                groups[True].append(record)
            else:
                groups[False].append(record)
        return groups
    def calculate_information_gain(self, data, attribute, threshold):
        """ Calculates the information gain for numerical threshold.
            :param data: training data
            :type data: list
            :param attribute: numerical attribute used for splitting
            :type attribute: str
            :param threshold: value used to split the records
            :type threshold: float
            :returns: information gain for the attribute
            :rtype: float
        """
        # Calculate the entropy of the original dataset
        original_entropy = self.calculate_entropy(data, "Class")

        # Split the data 
        groups = self.split_data(data, attribute, threshold)

        # Calculate weighted entropy
        weighted_entropy = self.calculate_weighted_entropy(groups)

        # Calculate information gain
        information_gain = original_entropy - weighted_entropy

        return information_gain
    def find_best_split(self, data, attribute):
        """ Finds the threshold with the highest information gain.
            :param data: training data
            :type data: list
            :param attribute: numerical attribute used for splitting
            :type attribute: str
            :returns: best threshold and its information gain
            :rtype: tuple
        """
        values = sorted(set(record[attribute] for record in data))

        if len(values) < 2:
            return None, 0  # Not enough unique values to split

        best_threshold = None
        best_gain = float('-inf')

        for i in range(len(values) - 1):
            threshold = (values[i] + values[i + 1]) / 2
            gain = self.calculate_information_gain(data, attribute, threshold)

            if gain > best_gain:
                best_gain = gain
                best_threshold = threshold
        return best_threshold, best_gain
    def build_tree(self, data, attributes):
        """ Builds a decision stump (one-level decision tree).
            :param data: training data
            :type data: list
            :param attributes: list of attributes to consider for splitting
            :type attributes: list
            :returns: root node of the decision stump
            :rtype: dict
        """
        if not data:
            raise ValueError("Data cannot be empty.")
        if self.is_pure(data) or not attributes:
            return {"class": self.find_majority_class(data)}

        attribute = attributes[0]  # Only consider the first attribute for a decision stump
        threshold, information_gain = self.find_best_split(data, attribute)

        if threshold is None or information_gain <= 0:
            return {"class": self.find_majority_class(data)}

        groups = self.split_data(data, attribute, threshold)

        node = {
            "attribute": attribute,
            "threshold": threshold,
            "majority_class": self.find_majority_class(data),
            "branches":{}
        }

        # Both children are leaves: a stump never splits again.
        for branch, group in groups.items():
            node["branches"][branch] = {"class": self.find_majority_class(group)}
        return node
    def _predict_node(self, node, instance):
        """ Predicts a class using the numerical slit.
            :param node: current node
            :type node: dict
            :param instance: record to classify
            :type instance: dict
            :returns: predicted class
            :rtype: int
        """
        if "class" in node:
            return node["class"]

        attribute = node["attribute"]
        threshold = node["threshold"]
        branch = instance[attribute] <= threshold

        return node["branches"][branch]["class"]
    def _draw_node(self, graph, node, node_id):
        """ Draws the threshold and class leaves for the stump.
            :param graph: graph object
            :type graph: Graph
            :param node: current node
            :type node: dict
            :param node_id: current node id
            :type node_id: int
            :returns: None
        """
        if "class" in node:
            graph.node(node_id, label=str(node["class"]), shape="ellipse")
            return
        label = f"{node["attribute"]} <= {node["threshold"]:g}"
        graph.node(node_id, label=label, shape="box")

        for branch, child in node["branches"].items():
            child_id = f"{node_id}_{branch}"
            self._draw_node(graph, child, child_id)
            graph.edge(node_id, child_id, label="True" if branch else "False")