
from template import NaiveBayesClassifier
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import spacy
from spacy.lang.en import English

def tokenize_and_lemmatize(text):
    tokens = word_tokenize(text.lower()) 

    lemmatizer = WordNetLemmatizer()

    tokens = [lemmatizer.lemmatize(token) for token in tokens if token.isalnum() and token not in stopwords.words('english')]

    return tokens
def preprocess(tweet_string):
    # clean the data and tokenize it
    tokens = tokenize_and_lemmatize(tweet_string)
    # print(f"Tokens after lemmatization: {tokens}")
    return tokens
def load_data(data_path):
    df = pd.read_csv(data_path)
    df = df.dropna(subset=['text'])  

    print("Labels before filtering:", set(df['label']))

    # Ensure 'label' column has the expected format
    # valid_labels = {0, 1, 2}
    # df = df[df['label'].isin(valid_labels)]

    print("Labels after filtering:", set(df['label']))

    return [(preprocess(tweet), label) for tweet, label in zip(df['text'], df['label'])]

# train your model and report the duration time
train_data_path = "train_data.csv" 
eval_data_path = "eval_data.csv"
main_test_data_path = "test_data_nolabel.csv"
classes = [0, 1, 2]
nb_classifier = NaiveBayesClassifier(classes)
nb_classifier.train(load_data(train_data_path))
data_for_training = load_data(train_data_path)

print("Classes:", nb_classifier.classes)
print("Vocabulary:", nb_classifier.vocab)
# print("Word Probabilities:", nb_classifier.word_probabilities)

# print(data_for_training)
# test_string = "i adore it "
# print(preprocess(test_string))
# classification_result = nb_classifier.classify(preprocess(test_string))
# print("Classification Result:", classification_result)
eval_data = load_data(eval_data_path)
# main_test_data = load_data(main_test_data_path)
# Open a text file for writing results
with open("evaluation_results.txt", "w", encoding="utf-8") as eval_file:
    eval_results = []  
    val_results = []  
    correct_predictions = 0
    incorrect_predictions = 0
    true_labels = []
    predicted_labels = []
    
    class_metrics = {}
    
    for tweet, actual_label in eval_data:
        eval_file.write(f"Processing tweet: {tweet}\n")
        print(f"Original tweet: {tweet}")
        # Preprocess the tweet using your preprocess function
        # processed_tweet = preprocess(tweet) 
        predicted_label = nb_classifier.classify(tweet)
    
        eval_results.append((predicted_label, actual_label))
        val_results.append((tweet, predicted_label, actual_label))
    
        eval_file.write(f"Processed: {tweet}\n")
        eval_file.write(f"Predicted: {predicted_label}, Actual: {actual_label}\n")
        eval_file.write("===\n")
    
        true_labels.append(actual_label)
        predicted_labels.append(predicted_label)
    
        if predicted_label == actual_label:
            correct_predictions += 1
        else:
            incorrect_predictions += 1
    
        if actual_label not in class_metrics:
            class_metrics[actual_label] = {'true': 0, 'false': 0}
    
        if predicted_label == actual_label:
            class_metrics[actual_label]['true'] += 1
        else:
            class_metrics[actual_label]['false'] += 1
    
    total_instances = len(eval_results)
    accuracy = correct_predictions / total_instances
    
    class_accuracies = {}
    for label in class_metrics:
        true_predictions = class_metrics[label]['true']
        total_instances_class = true_predictions + class_metrics[label]['false']
        class_accuracies[label] = true_predictions / total_instances_class
    
    class_precisions = {}
    class_recalls = {}
    class_f1_scores = {}
    
    for label in class_metrics:
        label_true_labels = [1 if true_label == label else 0 for true_label in true_labels]
        label_predicted_labels = [1 if predicted_label == label else 0 for predicted_label in predicted_labels]
    
        class_precisions[label] = precision_score(label_true_labels, label_predicted_labels)
        class_recalls[label] = recall_score(label_true_labels, label_predicted_labels)
        class_f1_scores[label] = f1_score(label_true_labels, label_predicted_labels)
    
    for label in class_metrics:
        eval_file.write(f"\nMetrics for Class '{label}':\n")
        eval_file.write(f"Accuracy: {class_accuracies[label]:.2%}\n")
        eval_file.write(f"True Predictions: {class_metrics[label]['true']}\n")
        eval_file.write(f"False Predictions: {class_metrics[label]['false']}\n")
        eval_file.write(f"Precision: {class_precisions[label]:.2f}\n")
        eval_file.write(f"Recall: {class_recalls[label]:.2f}\n")
        eval_file.write(f"F1 Score: {class_f1_scores[label]:.2f}\n")
    
    eval_file.write(f"\nOverall Metrics:\n")
    eval_file.write(f"Overall Accuracy: {accuracy:.2%}\n")
    eval_file.write(f"Correct Predictions: {correct_predictions}\n")
    eval_file.write(f"Incorrect Predictions: {incorrect_predictions}\n")

# with open("result.txt", "w", encoding="utf-8") as result_file:
#     for tweet, _ in main_test_data:
#         # result_file.write(f"Original tweet: {tweet}\n")

#         # Preprocess the tweet using your preprocess function
#         # processed_tweet = preprocess(tweet)

#         # Classify the tweet using your NaiveBayesClassifier
#         predicted_label = nb_classifier.classify(tweet)

#         # Write the predicted label to the result file
#         result_file.write(str(predicted_label))
#         result_file.write("\n")
