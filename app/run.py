'''
The web app enables the user to enter a disaster message, and then view the categories of the message.
The main page includes at least two visualizations using data from the SQLite database.
When a user inputs a message into the app, the app returns classification results for all 36 categories.

'''

import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine("sqlite:///../data/DisasterResponse.db")
df = pd.read_sql_table(table_name="DisasterDatabase", con=engine)

# load model
model = joblib.load("../models/classifier.pkl")

# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # Graphic 1
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    # Graphic 2
    df_sub = df.drop(columns=['id', 'message', 'genre'])
    col = df_sub.columns

    counts_array = []
    for i in range(0, df_sub.shape[1]):
        print(col[i], sum(df_sub.iloc[:, i].astype(int)))
        counts_array.append(sum(df_sub.iloc[:, i].astype(int)))
    df_counts = pd.DataFrame(index=col, data=counts_array, columns=["categories"]).sort_values(by=["categories"],                                                                                    ascending=False)

    total_counts = df_counts["categories"]
    total_names = list(df_counts.index)

    # create visuals
    graphs = [
        # Graphic 1
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },

        # Graphic 2
        {
            'data': [
                Bar(
                    x=total_names,
                    y=total_counts
                )
            ],

            'layout': {
                'title': 'Most frequent categories',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Categories"
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()