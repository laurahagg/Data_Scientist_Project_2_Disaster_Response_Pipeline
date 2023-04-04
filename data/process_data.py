import sys

# import libraries
import pandas as pd
import sqlite3
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):

    """
    Loads the Data

    Arguments:
        messages_filepath: csv file
        categories_filepath: csv file
    Output:
        df: merged dataframe from the two csv files.

    """

    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, on='id')
    return df


def clean_data(df):


    """
    Cleaning the Data

    Arguments:
        df: loaded dataframe
    Output:
        df: cleaned dataframe

    """
    categories = df['categories'].str.split(';', expand=True)
    row = categories.iloc[0]
    category_colnames = row.apply(lambda x: x[0:-2])

    # rename the columns of `categories`
    categories.columns = category_colnames

    for column in categories:
    # set each value to be the last character of the string
        categories[column] = categories[column].str[-1]
    # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])

    # drop the original categories column from `df`
    df=df.drop('categories', axis=1)

    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)

    # drop duplicates
    df.drop_duplicates(inplace=True)

    # convert all numbers to binary
    df['related'] = df['related'].astype('str').str.replace('2', '1')
    df['related'] = df['related'].astype('int')
    return df


def save_data(df, database_filename):
    """
    Save the Data
    Arguments:
        df: dataframe
        database_filename: how to name the database
    Output:
        None
    """
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('Messages', engine, index=False, if_exists='replace')
    pass


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
