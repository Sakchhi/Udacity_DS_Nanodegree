import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    Load data from csv files at given filepath
    :param messages_filepath: Filepath for messages csv file
    :param categories_filepath: Filepath for categories csv file
    :return: Dataframe merging both the csv files
    """
    df_categories = pd.read_csv(categories_filepath)
    df_messages = pd.read_csv(messages_filepath)
    df = pd.merge(df_categories, df_messages, on='id')
    return df


def clean_data(df):
    """
    Cleans Dataframe by converting Response categories into separate columns with binary values
    :param df: Dataframe containing responses and categories
    :return: Cleaned Dataframe
    """
    categories = df.categories.str.split(';', expand=True)

    row = categories.iloc[0, :]
    category_columns = row.apply(lambda r: r.split('-')[0])
    categories.columns = category_columns

    for column in categories:
        categories[column] = categories[column].apply(lambda r: r.split('-')[1].strip())
        categories[column] = categories[column].astype(int)

    df.drop(columns=['categories'], inplace=True)
    df = pd.merge(df, categories, left_index=True, right_index=True)

    df = df.drop_duplicates()
    return df


def save_data(df, database_filename):
    """
    Save dataframe to sqlite database at specified path
    :param df: Dataframe to be saved to sqlite database
    :param database_filename: Path and name of database to save dataframe to
    :return: None
    """
    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql('disaster_response_etl', engine, index=False, if_exists='replace')


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
        print(df.head())
        
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