'''
The script takes the file paths of the two datasets and database, cleans the datasets, and stores the clean data into a
SQLite database in the specified database file path.
'''

import pandas as pd
import sys
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    '''
    This function loads the messages and categories csv files.
    Afterwards, it merges them on the column id.

    :param messages_filepath: path cantaining the messages csv file
    :param categories_filepath: path cantaining the categories csv file
    :return: merged data frame
    '''

    df_messages   = pd.read_csv(messages_filepath,   encoding='latin-1')
    df_categories = pd.read_csv(categories_filepath, encoding='latin-1')

    # merge both datasets on id
    df = pd.merge(df_categories, df_messages, how='left', on='id')

    return df


def clean_data(df):
    '''
    This function cleans the data by:
        - splitting the categories column into separate, clearly named columns
        - dropping nan columns
        - dropping duplicates

    :param df: pandas dataframe to clean
    :return: cleaned pandas dataframe
    '''

    # number of catefories
    cat_number = df['categories'].str.split(pat = ';', expand = True).shape[1]

    for i in range(0, cat_number):
        # get column name
        col_name = df['categories'].str.split(';').str.get(i).unique()[0].split('-')[0]

        # parse to a int value
        df.loc[:, col_name] = df['categories'].str.split(';').str.get(i).str.split('-').str.get(1)

        if (len(df[col_name].unique()) == 1):
            print('column {} is always {}'.format(col_name, df[col_name].unique()))

    df.drop(columns='categories', inplace=True)

    # drop original value since it has many nan's (shall be identical to message??)
    df.drop(columns='original', inplace=True)
    #print('Nan\'s: ', df.isnull().sum().mean())

    # remove duplicated ids
    df = df[df.duplicated(subset=['id'], keep='first') == False]

    # duplicated messages (?) remove them
    df = df[df.duplicated(subset=['message'], keep='first') == False]

    return df


def save_data(df, database_filename):
    '''
    This function saves the cleaned data to a database

    :param df: Pandas dataframe to save
    :param database_filename: Database file name
    '''
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql(name = 'DisasterDatabase', con=engine, index=False, if_exists='replace')


# messages_filepath = 'data\messages.csv'
# categories_filepath = 'data\categories.csv'
# database_filepath = 'data\DisasterResponse.db'
# df = load_data(messages_filepath, categories_filepath)
# print('Cleaning data...')
# df = clean_data(df)
# print('Saving data...\n    DATABASE: {}'.format(database_filepath))
# save_data(df, database_filepath)
# print('Cleaned data saved to database!')

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

    #else:
        print('Please provide the filepaths of the messages and categories '
              'datasets as the first and second argument respectively, as ' 
              'well as the filepath of the database to save the cleaned data ' 
              'to as the third argument. \n\nExample: python process_data.py ' 
              'disaster_messages.csv disaster_categories.csv ' 
              'DisasterResponse.db')

if __name__ == '__main__':
    main()


