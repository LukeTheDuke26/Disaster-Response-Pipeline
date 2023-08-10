import os
import sys
import traceback
import pandas as pd
import nltk
import pickle
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

nltk.download(['punkt', 'wordnet'])


def load_data(database_filepath):
    """
    Load data from the SQLite database.
    
    Args:
    database_filepath: str. Filepath for SQLite database containing preprocessed data.
    
    Returns:
    X: pd.Series. Messages for training.
    Y: pd.DataFrame. Categories for training.
    category_names: list. Category names for evaluation.
    """
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('DisasterResponse', engine)
    X = df['message']
    Y = df.drop(['id', 'message', 'genre'], axis=1)
    category_names = list(Y.columns.values)
    return X, Y, category_names


def tokenize(text):
    """
    Tokenizes text data.
    
    Args:
    text: str. Messages for tokenization.
    
    Returns:
    clean_tokens: list. List of tokens extracted from the provided text.
    """
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(tok).lower().strip() for tok in tokens]


def build_model():
    """
    Build a machine learning pipeline.
    
    Returns:
    pipeline: sklearn.Pipeline. Machine learning pipeline.
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', RandomForestClassifier())
    ])
    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluate model performance.
    
    Args:
    model: sklearn.Pipeline. Trained model.
    X_test: pd.Series. Test messages.
    Y_test: pd.DataFrame. True categories for test messages.
    category_names: list. Category names for evaluation.
    
    Returns:
    misclassified_indices: list. Indices of misclassified messages for "related" category.
    """
    Y_pred = model.predict(X_test)
    misclassified_indices = []

    for i, col in enumerate(category_names):
        print(col)
        print(classification_report(Y_test[col], Y_pred[:, i]))

    related_col_index = category_names.index('related')
    y_true_related = Y_test['related'].values
    y_pred_related = Y_pred[:, related_col_index]

    for idx, (true, pred) in enumerate(zip(y_true_related, y_pred_related)):
        if true != pred:
            misclassified_indices.append(idx)

    return misclassified_indices


def save_model(model, model_filepath):
    """
    Save trained model as a pickle file.
    
    Args:
    model: sklearn.Pipeline. Trained model.
    model_filepath: str. Filepath for where to save the trained model.
    """
    try:
        with open(model_filepath, 'wb') as file:
            pickle.dump(model, file)
        print("Model saved successfully.")
    except Exception as e:
        print("Failed to save model.")
        print(e)
        print(traceback.format_exc())


def main():
    """
    Main function for running the script.
    """
    print("Current working directory:", os.getcwd())
    database_filepath = '/Users/luca/Documents/Udacity - all learning materials/disaster_response_pipeline_project/data/DisasterResponse.db'
    model_filepath = '/Users/luca/Documents/Udacity - all learning materials/disaster_response_pipeline_project/data/classifier.pkl'

    print('Loading data...\n    DATABASE: {}'.format(database_filepath))
    X, Y, category_names = load_data(database_filepath)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

    print('Building model...')
    model = build_model()

    print('Training model...')
    model.fit(X_train, Y_train)

    print('Evaluating model...')
    misclassified_indices = evaluate_model(model, X_test, Y_test, category_names)

    print('\nMisclassified Messages for "related" category:')
    for idx in misclassified_indices[:10]:
        print(X_test.iloc[idx])

    print('Saving model...\n    MODEL: {}'.format(model_filepath))
    save_model(model, model_filepath)

    print('Trained model saved!')


if __name__ == '__main__':
    main()
