# Created by tnrbf@umsystem.edu on 2026-11-19

# Last updated by tnrbf@umsystem.edu on 2026-11-120

from Decision_Stump import Decision_Stump

import math
import random

class AdaBoost:
    # Constructor
    def __init__(self, max_rounds=100, random_seed=42):
        """ Constructs an AdaBoost classifier.
        :param max_rounds: maximum number of boosting rounds
        :type max_rounds: int
        :param random_seed: seed for random number generator
        :type random_seed: int
        :rtype: None 
        """
        # Data fields
        self._models = []  # List to store weak classifiers
        self._alphas = []  # List to store alpha values for each weak classifier
        self._probabilities = []  # List to store probabilities for each weak classifier
        self._max_rounds = max_rounds
        self._round_results = []  # List to store results of each boosting round
        self._stopping_reason = None  # Reason for stopping the boosting process
        self._random_seed = random_seed
        self._random_generator = random.Random(random_seed)  # Random number generator for reproducibility
    # Methods
    def initialize_probabilities(self, data):
        """ Initializes an equal selection probability for each record
            :param data: training data
            :type data: list
            :rturns: initial probability distribution
            :rtype: list
        """
        if not data:
            raise ValueError("Data cannot be empty.")

        probability = 1.0 / len(data)
        self._probabilities = [probability for record in data]

        return self._probabilities.copy()  # Return a copy to avoid external modification
    def sample_records(self, data, probabilities):
        """ Samples training records with replacement using probabilities.
            :param data: training data
            :type data: list    
            :param probabilities: selection probabilities for each record
            :type probabilities: list
            :returns: sampled records
            :rtype: list
        """
        if len(data) == 0 or len(data) != len(probabilities):
            raise ValueError("Data and probabilities must be of the same non-zero length.")
        if any(probability < 0 for probability in probabilities):
            raise ValueError("Probabilities must be non-negative.")
        if not math.isclose(sum(probabilities), 1.0):
            raise ValueError("Probabilities must sum to 1.")

        sampled_data = self._random_generator.choices(data, weights=probabilities, k=len(data))

        return sampled_data
    def calculate_error(self, model, data, probabilities):
        """ Calculates the weighted error of model.
            :param model: trained decision stump model
            :type model: Decision_Stump
            :param data: training data
            :type data: list
            :param probabilities: current probability distribution
            :type probabilities: list
            :returns: weighted error of the model
            :rtype: float
        """
        error = 0.0

        for i, record in enumerate(data):
            prediction = model.predict(record)
            actual = record["Class"]
            if prediction != actual:
                error += probabilities[i]

        return error
    def calculate_alpha(self, error):
        """ Calculates the weight of a weak classifier.
            :param error: weighted error of the weak classifier
            :type error: float
            :returns: alpha value
            :rtype: float
        """
        if error <= 0 or error >= 0.5:
            raise ValueError("Error must be in the range (0, 0.5) to calculate alpha.")

        alpha = 0.5 * math.log((1.0 - error) / error)

        return alpha
    def update_probabilities(self, model, data, probabilities, error):
        """ Updates the selection probabilities of training records.
            :param model: trained decision stump model
            :type model: Decision_Stump
            :param data: training data
            :type data: list
            :param probabilities: current probability distribution
            :type probabilities: list
            :param error: weighted error of the weak classifier
            :type error: float
            :returns: updated probability distribution
            :rtype: list
        """
        updated_probabilities = []

        for i, record in enumerate(data):
            prediction = model.predict(record)
            actual = record["Class"]
            if prediction != actual:
                probability = probabilities[i] / (2 * error)
            else:
                probability = probabilities[i] / (2 * (1 - error))
            updated_probabilities.append(probability)
        self._probabilities = updated_probabilities

        return updated_probabilities.copy()
    def reset(self):
        """ Resets the AdaBoost classifier to its initial state.
            :rtype: None
        """
        self._models = []
        self._alphas = []
        self._round_results = []
        self._stopping_reason = None
        self._random_generator = random.Random(self._random_seed)  # Reset the random generator for reproducibility
    def train(self, data, attributes):
        """ Trains the AdaBoost classifier.
            :param data: training data
            :type data: list
            :param attributes: list of attribute names
            :type attributes: list
            :rtype: None
        """
        if not data:
            raise ValueError("Training data cannot be empty.")

        # Resets the ensemble before training
        self.reset()

        # Initialize probabilities
        probabilities = self.initialize_probabilities(data)

        # Maximum number of attempts for each boosting round
        max_attempts = 100

        for t in range(1, self._max_rounds + 1):
            attempt = 0
            while attempt < max_attempts:
                attempt += 1

                # Sample training records
                sampled_data = self.sample_records(data, probabilities)

                # Train a decision stump
                model = Decision_Stump()
                model.train(sampled_data, attributes)

                # Calculate weighted classification error
                error = self.calculate_error(model, data, probabilities)

                # Stop if the weak classifier is perfect
                if math.isclose(error, 0.0, abs_tol=1e-15):
                    self._models.append(model)
                    self._alphas.append(1.0)  # Assign maximum weight to a perfect classifier

                    self._round_results.append({
                        "round": t,
                        "probabilities": probabilities.copy(),
                        "sampled_data": sampled_data.copy(),
                        "model": model,
                        "error": 0.0,
                        "alpha": float('inf'),  # Infinite alpha for perfect classifier
                    })
                    self._stopping_reason = "Perfect classifier"
                    return  # Stop training as we have a perfect classifier

                # Reject a weak classifier with error >= 0.5
                if error >= 0.5:
                    probabilities = self.initialize_probabilities(data)  # Reset probabilities
                    continue  # Retry training a new weak classifier

                # Calculate the weight of the weak classifier
                alpha = self.calculate_alpha(error)

                # Save the model and its weight
                self._models.append(model)
                self._alphas.append(alpha)

                # Store the results of this boosting round
                self._round_results.append({
                    "round": t,
                    "probabilities": probabilities.copy(),
                    "sampled_data": sampled_data.copy(),
                    "model": model,
                    "error": error,
                    "alpha": alpha,
                })

                # Update probabilities for the next round
                probabilities = self.update_probabilities(model, data, probabilities, error)

                # Current round completed successfully
                break  # Exit the while loop and proceed to the next boosting round
            else:
                # Failed to find a suitable weak classifier after max_attempts
                self._stopping_reason = "Maximum attempts reached"
                return
        self._stopping_reason = "Maximum boosting rounds reached"
    def calculate_ensemble_score(self, instance):
        """ Calculates the weighted sum of weak classifier predictions.
            :param instance: record to classify
            :type instance: dict
            :returns: weighted sum of predictions
            :rtype: float
        """
        if not self._models:
            raise ValueError("The AdaBoost classifier has not been trained yet.")

        # A perfect weak classifier overrides the ensemble models
        if self._stopping_reason == "Perfect classifier":
            return float(self._models[-1].predict(instance))  # Return the prediction of the perfect classifier

        score = 0.0
        for model, alpha in zip(self._models, self._alphas):
            prediction = model.predict(instance)
            score += alpha * prediction
        return score
    def predict(self, instance):
        """ Predicts the class of an instance using AaBoost ensemble.
            :param instance: record to classify
            :type instance: dict
            :returns: predicted class
            :rtype: int
        """
        score = self.calculate_ensemble_score(instance)

        if score > 0:
            return 1
        elif score < 0:
            return -1
        else:
            return 1 # Tie-breaking rule: predict positive class when score is zero
    def print_round_results(self):
        """ Prints the results of each boosting round.
            :rtype: None
        """
        if not self._round_results:
            print("No boosting results available.")
            return

        for result in self._round_results:
            round_num = result["round"]
            probabilities = result["probabilities"]
            sampled_data = result["sampled_data"]
            model = result["model"]
            error = result["error"]
            alpha = result["alpha"]

            print(f"\nBoosting Round {round_num}:")
            print('-' * 70)

            print("Probabilities:")
            print([round(prob, 4) for prob in probabilities])

            print("\nSampled Records:")

            print("X:", [record['X'] for record in sampled_data])
            print("Y:", [record['Class'] for record in sampled_data])

        root = model.get_root()
        print("\nDecision Stump:")

        if "class" in root:
            print(f"Predict Class: {root['class']}")
        else:
            threshold = root["threshold"]
            left_class = root["branches"][True]["class"]
            right_class = root["branches"][False]["class"]

            print(f"X <= {threshold:g} ==> Class = {left_class}")
            print(f"X > {threshold:g} ==> Class = {right_class}")
        print(f"\nEpsilon = {error:.6f}")
        print(f"Alpha = {alpha:.6f}")
    def print_probabilities(self, data):
        """ Prints the probability distribution for each boosting round.
            :param data: training data
            :type data: list
            :rtype: None
        """
        if not self._round_results:
            print("No boosting results available.")
            return
        print("\nProbability Distributions:")
        print('-' * 90)

        print(f"{'Round':<8}", end="")

        # Print Table Header
        for record in data:
            print(f"{record['X']:<8.2f}", end="")
        print()

        # Print the probabilities for each boosting round
        for result in self._round_results:
            print(f"{result['round']:<8}", end="")
            for prob in result["probabilities"]:
                print(f"{prob:<8.4f}", end="")
            print()
    def print_model_summary(self):
        """ Prints a summary of all trained decision stump.
            :rtype: None
        """
        if not self._round_results:
            print("No boosting results available.")

        print("\nBoosting Model Summary:")
        print('-' * 75)

        print(f"{'Round':<8}{'Threshold':<12}{'Left':<10}"
              f"{'Right':<10}{'Epsilon':<12}{'Alpha':<12}")

        for result in self._round_results:
            root = result["model"].get_root()

            if "class" in root:
                threshold = "N/A"
                left_class = root["class"]
                right_class = root["class"]
            else:
                threshold = f'{root["threshold"]:.4f}'
                left_class = root["branches"][True]["class"]
                right_class = root["branches"][False]["class"]

            print(f'{result["round"]:<8}'
                  f'{threshold:<12}'
                  f'{left_class:<10}'
                  f'{right_class:<10}'
                  f'{result["error"]:<12.6f}'
                  f'{result["alpha"]:<12.6f}')
        print("-" * 75)
        print(f"Total Boosting Round: {len(self._models)}")
        print(f"Stopping Reason: {self._stopping_reason}")
    def print_ensemble_results(self, data):
        """ Prints the ensemble classification results.
            :param data: data to classify
            :type data: list
            :rtype: None
        """
        if not self._models:
            print("No boosting results available.")
            return

        print("\nEnsemble Classification Results:")
        print('-' * 100)

        # Print Table Header
        print(f"{'Round':<8}", end="")
        for record in data:
            print(f"{record['X']:<9.2f}", end="")
        print()

        # Print predictions from each decision stump
        for t, model in enumerate(self._models, start=1):
            print(f"{t:<8}", end="")
            for record in data:
                prediction = model.predict(record)
                print(f"{prediction:<9}", end="")
            print()

        # Print the weighted sums
        print(f"{'Sum':<8}", end="")
        for record in data:
            score = self.calculate_ensemble_score(record)
            print(f"{score:<9.3f}", end="")
        print()

        # Print the final ensemble predictions
        print(f"{'Sign':<8}", end="")
        for record in data:
            prediction = self.predict(record)
            print(f"{prediction:<9}", end="")
        print()
    def calculate_training_error(self, data):
        """ Calculates the classification error.
            :param data: training data
            :type data: list
            :returns: The calculated classification error rate
            :rtype: float
        """
        if not data:
            raise ValueError("Data cannot be empty.")
        
        incorrect = 0
        for record in data:
            prediction = self.predict(record)
            if prediction != record["Class"]:
                incorrect += 1

        training_error = incorrect / len(data)

        return training_error




