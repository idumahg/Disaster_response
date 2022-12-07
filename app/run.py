import json
import plotly
import pandas as pd
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
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
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('DisasterResponseTable', engine)
y = df[list(set(df.columns) - set(df[['id', 'message', 'genre', 'original']]))]

# load model
model = joblib.load("../models/classifier.pkl")

def return_figures():
    """
    Creates two plotly visualizations.

    Parameters:
        None.

    Returns:
        list (dict): list containing the two visualizations.
    """
    # First bar chart of Genre Count
    graph_one = []
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    graph_one.append(
        Bar(
            x = genre_names,
            y = genre_counts,
        )
    )

    layout_one = dict(title = 'Distribution of Message Genres',
                      xaxis = dict(title = 'Genre'),
                      yaxis = dict(title = 'Count'),)


    # Second bar chart of Category percentage
    graph_two = []
    category_perc = y.mean() * 100
    category_names = list(y.columns)

    graph_two.append(
        Bar(
            x=category_names,
            y=category_perc,
        )
    )

    layout_two = dict(title='Percentage of category in data',
                      xaxis=dict(title='Categories'),
                      yaxis=dict(title='Percentage'), )

    # append all charts to the figures list
    figures = [dict(data=graph_one, layout=layout_one), dict(data=graph_two, layout=layout_two)]

    return figures


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():

    figures = return_figures()

    # plot ids for the html id tag
    ids = ["graph-{}".format(i) for i, _ in enumerate(figures)]

    # Convert the plotly figures to JSON for javascript in html template
    figuresJSON = json.dumps(figures, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, figuresJSON=figuresJSON)


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
    app.run(host='0.0.0.0', port=3000, debug=True)


if __name__ == '__main__':
    main()