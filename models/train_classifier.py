'''
The script takes the database file path and model file path, creates and trains a classifier, and stores the classifier
into a pickle file to the specified model file path.

This file:
 Loads data from the SQLite database
 Splits the dataset into training and test sets
 Builds a text processing and machine learning pipeline that uses NLTK, scikit-learn's Pipeline and GridSearchCV
 Trains and tunes a model using GridSearchCV
 Outputs results on the test set: a final model that uses the message column to predict classifications for 36 categories
 Exports the final model as a pickle file
'''

# import libraries
import sys
from sqlalchemy import create_engine
import pandas as pd
import re
import pickle
import numpy as np

import nltk
import warnings
warnings.simplefilter('ignore')
warnings.filterwarnings('ignore')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import GridSearchCV


def load_database(database_filepath):
    '''
    This function load the message database.

    :param database_filepath: filepath of the disaster messages database
    :return: X: text message column
             y: categories
             category_names: name of each category column
    '''
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table(table_name="DisasterDatabase", con=engine)
    X = df['message']
    y = df.iloc[:,3:]
    category_names = y.columns.tolist()

    return X, y, category_names

#load_database()


def tokenize(text):
    '''
    This function tokenizes the text data from the database messages.

    :param text: text data
    :return: cleaned and tokenized text
    '''

    # Remove punctuation
    text = re.sub(r'[^a-zA-Z0-9]', ' ', text)

    # tokenize text
    tokens = word_tokenize(text)

    # Remove stop words
    #tokens_short = [w for w in tokens if w not in stopwords.words("english")]

    # initiate lemmatizer
    lemmatizer = WordNetLemmatizer()

    # iterate through each token
    clean_tokens = []
    for tok in tokens:
        # lemmatize, normalize case, and remove leading/trailing white space
        clean_tok = lemmatizer.lemmatize(tok, pos='n').lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    '''
    It builds the machine pipeline should take in the `message` column as input and output classification results on
    the other 36 categories in the dataset.
    For more information see MultiOutputClassifier at
    http://scikit-learn.org/stable/modules/generated/sklearn.multioutput.MultiOutputClassifier.html

    :return: the pipeline
    '''

    # build pipeline
    pipeline = Pipeline([
        # first steps are transformers
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        # last step is an estimator
        # SGDClassifier, XGBoost or AdaBoost are other possible classifier estimators
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    parameters = {
        'tfidf__norm': ['l1','l2'],
        'clf__estimator__n_estimators': [10, 25]
    }

    cv = GridSearchCV(pipeline, param_grid=parameters, n_jobs=-1, verbose=1)

    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    This function shows the accuracy, precision, and recall of the tuned model.
    '''
    Y_pred = model.predict(X_test)

    # Report the f1 score, precision and recall for each output category of the dataset. You can do this by iterating
    # through the columns and calling sklearn's `classification_report` on each.
    for i in range(len(category_names)):
        print("Label:", category_names[i])
        print(classification_report(Y_test.values[:, i], Y_pred[:, i]))


def save_model(model, model_filepath):
    '''
    This function exports the model as pickle file

    :param model: model to be saved
    :param model_filepath: path to save
    '''

    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)



def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_database(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '
              'as the first argument and the filepath of the pickle file to '
              'save the model to as the second argument. \n\nExample: python '
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')

if __name__ == '__main__':
    main()



