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
It is included in the `process_data.py` file. In this script the data is cleaned and stored in a _SQlite_ database.

### 2. ML pipeline
It is included in the `train_classifier.py` file.

## Instructions
1. Run ETL pipeline: `python process_data.py data/messages.csv data/categories.csv data/DisasterResponse.db`
2. Run ML pipeline: `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`
3. Run the web app: `python run.py`
4. Go to: [http://localhost:3001](http://localhost:3001)

## Acknowledgements
This project is part of the [Udacity Data Analysis Nanodegree](https://www.udacity.com/course/data-scientist-nanodegree--nd025).