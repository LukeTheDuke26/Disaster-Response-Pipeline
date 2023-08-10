import json
import plotly
import pandas as pd
import string

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize, RegexpTokenizer

from flask import Flask, render_template, request, jsonify
from plotly.graph_objs import Bar
from collections import Counter
from nltk.corpus import stopwords
import joblib
from sqlalchemy import create_engine

app = Flask(__name__)


def tokenize(text):
    """Normalize, tokenize and lemmatize text."""
    
    # Tokenize using a regular expression tokenizer to exclude punctuation
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(text)
    
    lemmatizer = WordNetLemmatizer()

    # Exclude stopwords from tokens and lemmatize
    stop_words = set(stopwords.words('english'))
    clean_tokens = [lemmatizer.lemmatize(tok).lower().strip() for tok in tokens if tok.lower() not in stop_words]

    return clean_tokens


def compute_word_frequencies(df):
    """Compute word frequencies for messages in DataFrame.
    
    Args:
    - df: DataFrame containing messages to be analyzed.

    Returns:
    - words (list): Top 10 frequent words.
    - counts (list): Counts of the top 10 frequent words.
    """
    
    stop_words = set(stopwords.words('english'))
    all_words = []
    for message in df['message']:
        tokens = tokenize(message)
        all_words.extend([token for token in tokens if token not in stop_words])
    
    word_freq = Counter(all_words)
    sorted_word_freq = word_freq.most_common(10)
    words = [word[0] for word in sorted_word_freq]
    counts = [word[1] for word in sorted_word_freq]
    
    return words, counts


# Load data
engine = create_engine('sqlite:////Users/luca/Documents/Udacity - all learning materials/disaster_response_pipeline_project/data/DisasterResponse.db')
df = pd.read_sql_table('DisasterResponse', engine)
words, counts = compute_word_frequencies(df)

# Data for category visualization
category_counts = df.iloc[:, 4:].sum().sort_values(ascending=False)
category_names = list(category_counts.index)

# Load model
model = joblib.load("/Users/luca/Documents/Udacity - all learning materials/disaster_response_pipeline_project/data/classifier.pkl")


@app.route('/')
@app.route('/index')
def index():
    """Render main page with plots."""
    
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    # Original Genre Distribution Visualization
    genre_graph = {
        'data': [Bar(x=genre_names, y=genre_counts)],
        'layout': {
            'title': 'Distribution of Message Genres',
            'yaxis': {'title': "Count"},
            'xaxis': {'title': "Genre"}
        }
    }

    # New Category Distribution Visualization
    category_graph = {
        'data': [Bar(x=category_names, y=category_counts)],
        'layout': {
            'title': 'Distribution of Message Categories',
            'yaxis': {'title': "Count"},
            'xaxis': {'title': "Category"}
        }
    }

    # Word Frequency Visualization
    word_freq_graph = {
        'data': [Bar(x=words, y=counts)],
        'layout': {
            'title': 'Top 10 Words in Messages',
            'yaxis': {'title': "Count"},
            'xaxis': {'title': "Words"}
        }
    }

    graphs = [genre_graph, category_graph, word_freq_graph]
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    return render_template('master.html', ids=ids, graphJSON=graphJSON)


@app.route('/go')
def go():
    """Render page with classification results for user query."""
    
    query = request.args.get('query', '') 
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    return render_template('go.html', query=query, classification_result=classification_results)


def main():
    """Run the app."""
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
