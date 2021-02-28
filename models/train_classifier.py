'''
The script takes the database file path and model file path, creates and trains a classifier, and stores the classifier
into a pickle file to the specified model file path.

This file:
- uses a custom tokenize function using nltk to case normalize, lemmatize, and tokenize text. This function is used in
the machine learning pipeline to vectorize and then apply TF-IDF to the text.
- builds a pipeline that processes text and then performs multi-output classification on the 36 categories in the
dataset. GridSearchCV is used to find the best parameters for the model.
- The TF-IDF pipeline is only trained with the training data. The f1 score, precision and recall for the test set is
outputted for each category.

Steps:
 Loads data from the SQLite database
 Splits the dataset into training and test sets
 Builds a text processing and machine learning pipeline that uses NLTK, scikit-learn's Pipeline and GridSearchCV
 Trains and tunes a model using GridSearchCV
 Outputs results on the test set: a final model that uses the message column to predict classifications for 36 categories
 Exports the final model as a pickle file
'''

