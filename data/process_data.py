"""A code to implement ML pipeline."""
import sys
import pandas as pd
from sqlalchemy.engine import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    Load and merge csv files.

    Parameters:
        messages_filepath: messages.csv file path
        categories_filepath: categories.csv file path

    Returns:
        df: merged pandas df
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, how='inner', on='id')
    
    return df


def clean_data(df):
    """
    Clean data frame function.

    Parameters:
        df: unclean data frame

    Returns:
        df: clean data frame
    """
    categories = df.categories.str.split(";", expand = True) 
    row = categories.iloc[0]
    categories.columns = row.apply(lambda x: x.split('-')[0])
    
    # convert categories to just 0 or 1
    for column in categories:
        categories[column] = categories[column].apply(lambda x: x.split('-')[1])
        categories[column] = pd.to_numeric(categories[column])
        
    df.drop("categories", axis=1, inplace=True)
    df = pd.concat([df, categories], axis=1)
    
    # drop duplicate rows
    df = df.drop_duplicates()
    
     # make "related" column to be binary
    df.related.replace(2, 1, inplace=True)
    
    return df


def save_data(df, database_filename):
    """
    Save data to SQL database

    Parameters:
        df: pandas data frame
        database_filename: name of SQL database

    Returns:
        None
    """
    engine = create_engine('sqlite:///'+ database_filename)  
    df.to_sql('DisasterResponseTable', engine, index=False, if_exists='replace')


def main():
    """Script to run."""
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