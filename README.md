# Disaster-response-pipeline
After a disaster the social media will get millions of messages about what had happened and the immediate needs of the affected people.
Following the disaster response organizations will need to filter this data and take actions to specific problems.

## Motivation
The goal of the project is to analize disaster messages from [Figure Eight] (https://appen.com/) by building a
supervised learning model that classifies the messages into 36 pre-defined categories.

This project contains a web app in where a user can input a new message and the classification result will be displayed.
Moreover, it will display some visualization of the data.

## Installation
Th files contains python and HTML files. It requires Pythin version 3.* and the following packages: pandas, numpy,
pickle, re, nltk, sklear, sqlalchemy, sys, warnings, json, ploty and flask.

## Project components

### 1. ETL pipeline
It is included in the `process_data.py` file. The script takes the file paths of the two datasets and database, cleans the datasets, and stores the clean data into a
_SQLite_ database called _DisasterResponse.db_.

### 2. ML pipeline
It is included in the `train_classifier.py` file. This file:
 * Loads data from the _SQLite_ database
 * Splits the dataset into training and test sets
 * Builds a text processing and machine learning pipeline that uses NLTK, scikit-learn's Pipeline and GridSearchCV
 * Trains and tunes a model using GridSearchCV
 * Outputs results on the test set: a final model that uses the message column to predict classifications for 36 categories
 * Exports the final model as a pickle file called _classifier.pkl_

### 3. Web app
The web app enables the user to enter a disaster message, and then view in which of the 36 categories it is classified.
The main page includes two visualizations of the database in which the model has been trained. 

## Project structure

The files in the project follow the structure below:

* app
  * template
    * `master.html` - main page of web app
    * `go.html` - classification result page of web app
  * `run.py` - Flask file that runs app
* data
  * disaster_categories.csv - input data to process
  * disaster_messages.csv  - input data to process
  * 'process_data.py'
  * `DisasterResponse.db` - output database containing the clean data
* models
  * `train_classifier.py` - MPL model
  * `classifier.pkl` - saved model
*README.md

## Instructions
1. Run ETL pipeline: `python process_data.py data/messages.csv data/categories.csv data/DisasterResponse.db`
2. Run ML pipeline: `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`
3. Run the web app from the app folder: `python run.py`
4. Go to: [http://localhost:3001](http://localhost:3001)

## Acknowledgements
This project is part of the [Udacity Data Analysis Nanodegree](https://www.udacity.com/course/data-scientist-nanodegree--nd025).
