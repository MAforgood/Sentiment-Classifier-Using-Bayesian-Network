# Sentiment-Classifier-Using-Bayesian-Network

## Description
This repository contains the final project for the Artificial Intelligence Course Tu, Spring 2024. The project focuses on building a sentiment classifier based on a Bayesian network.


## Professor
- Dr. Mohammadreza Mohammadi
- 
## Group Members
- MohammadAli Khosorabadi

## Steps
### Step 1: Tokenization
- Tweets are tokenized using the NLTK library. Stopwords are removed, and tweets are lemmatized.

### Step 2: Loading Data
- Data is loaded using the Pandas library, followed by preprocessing operations on the dataset.

### Step 3: Training Our Model
- Implemented a Bayesian Network Model based on existing formulas.
  - `Calculate_Prior(self, label)`: Calculates the log prior probability of a class.
  - `Calculate_Likelihood(self, word, label)`: Calculates the log likelihood of a word given a class.
  - `Classify(self, features)`: Predicts the class for a given set of features.
- Training Time: Varies based on dataset size and computational resources.

### Step 4: Testing On Eval Dataset
- Model performance is evaluated on an evaluation dataset, providing overall accuracy, precision, recall, and accuracy for each class.
- Results are stored in `evaluation_result.txt`.

### Step 5: Testing On Test Dataset
- The model is applied to the test dataset, and predictions are stored in `result.txt`.

## Results
- Overall accuracy on the eval dataset: 63.94%.

## Conclusion
- The model shows reasonable performance considering the dataset used.
- Future improvements could include exploring additional methods and optimizing training time.


