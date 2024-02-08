# Naive Bayes 3-class Classifier 
# Authors: Baktash Ansari - Sina Zamani 

# complete each of the class methods  
import math
from collections import defaultdict
class NaiveBayesClassifier:

    def __init__(self, classes):
        # initialization: 
        # inputs: classes(list) --> list of label names
        # class_word_counts --> frequency dictionary for each class
        # class_counts --> number of instances of each class
        # vocab --> all unique words  
        self.classes = classes
        self.class_word_counts = {label: defaultdict(int) for label in classes}
        self.class_counts = {label: 0 for label in classes}
        self.vocab = set()

    def train(self, data):
        print("Initial Classes:", self.classes)
        print("Initial Vocabulary:", self.vocab)
        print("Labels in training data:", set(label for _, label in data))
        # Check if attributes are None and initialize them
        if self.classes is None:
            self.classes = set(label for _, label in data)
        if self.class_word_counts is None:
            self.class_word_counts = {label: defaultdict(int) for label in self.classes}
        if self.class_counts is None:
            self.class_counts = {label: 0 for label in self.classes}
        if self.vocab is None:
            self.vocab = set()

        # print("Classes after initialization:", self.classes)
        # print("Vocabulary after initialization:", self.vocab)
        for features, label in data:
            # print("Current features:", features)
            # print("Current label:", label)
            # Update word counts based on the label
            for word in features:
                self.class_word_counts[label][word] += 1
                self.vocab.add(word)
            self.class_counts[label] += 1


    def calculate_prior(self, label):
        # Calculate the prior probability of a class
        # inputs: label --> class label
        # output: log prior probability
        total_instances = sum(self.class_counts.values())
        
        if total_instances == 0:
             return 0.0
        
        prior = math.log(self.class_counts[label] / total_instances)
        return prior


    def calculate_likelihood(self, word, label):
        # Calculate the likelihood of a word given a class
        # inputs: word, label --> word and class label
        # output: log likelihood probability
        count_word_label = self.class_word_counts[label][word]
        count_label = sum(self.class_word_counts[label].values())

        V = len(self.vocab)

        likelihood = (count_word_label + 1) / (count_label + V)
        if (count_label+ V) == 0:
            return 0.0

        return math.log(likelihood)


    def classify(self, features):
        # predict the class
        # inputs: features(list) --> words of a tweet 
        predictions = {}


        for label in self.classes:
            log_prior = self.calculate_prior(label)
            log_likelihood_sum = sum(self.calculate_likelihood(word, label) for word in features)
            predictions[label] = log_prior + log_likelihood_sum
    
        print("***")
        print("Log Priors:", {label: self.calculate_prior(label) for label in self.classes})
        print("Log Likelihood Sums:", {label: sum(self.calculate_likelihood(word, label) for word in features) for label in self.classes})
        print("Predictions:", predictions)  
    
        # word_lambda_sum = sum(self.word_probabilities.get(word, {}).get('lambda_word', 0) for word in features)
    
        # for label in self.classes:
        #     predictions[label] += word_lambda_sum
    
        decision = max(predictions, key=predictions.get)
        
        return decision
    

# Good luck :)
