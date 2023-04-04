import sys

# import libraries
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import sqlite3

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

from sklearn.datasets import make_multilabel_classification
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier

import pickle
import os


def load_data(database_filepath):

    """
    Loads the Data

    Arguments:
        database_filepath -> Path to SQLite destination database (DisasterResponse.db)
    Output:
        X -> a dataframe with the features
        Y -> a dataframe with the  labels
        category_names -> List of the categories name
    """


    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('Messages', engine)

    X = df['message']
    Y = df.drop(['id', 'message', 'original', 'genre'], axis=1 )
    category_names = Y.columns.tolist()
    return X,Y, category_names


def tokenize(text):

    """
    Tokenizes the text

    Arguments:
        text = the messages to be tokenized
    Output:
        cleaned tokens
    """


    text=text.lower()
    tokens = word_tokenize(text)
    tokens = [WordNetLemmatizer().lemmatize(t) for t in tokens if t not in stopwords.words('english')]
    return tokens


def build_model():

    """
    Build the model

    Arguments:
        None
    Output:
        a pipeline of the model
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfifd', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
        ])

    parameters = {
    'clf__estimator__n_estimators':[10, 20],
    'clf__estimator__max_depth':[1,2]
    }

    model = GridSearchCV(pipeline, param_grid=parameters)
    return model


def evaluate_model(model, X_test, Y_test, category_names):
    """
    To Evaluate the Model


    Arguments:
        model -> the machine learning pipeline
        X_test -> Test features
        Y_test -> Test labels
        category_names -> names of the labels
    """
    Y_pred = model.predict(X_test)
    report = classification_report(Y_test, Y_pred, target_names= category_names)
    print(report)




def save_model(model, model_filepath):
    """
    Save the Data

    Arguments:
        model: trained model
        Model_filepath: filepath of the model
    Output:
        None

    """

    with open(model_filepath,'wb') as f:
        pickle.dump(model,f)
    pass


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
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
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
