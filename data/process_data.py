import argparse
import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    Load disaster messages and categories data from specified filepaths.
    
    Args:
    messages_filepath: string. Filepath for the messages dataset.
    categories_filepath: string. Filepath for the categories dataset.
    
    Returns:
    df: DataFrame. Merged dataset.
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    return messages.merge(categories, on='id')

def clean_data(df):
    """
    Clean the merged dataset to split categories into separate columns 
    and convert values to binary (0 or 1).
    
    Args:
    df: DataFrame. Merged dataset returned from load_data().
    
    Returns:
    df: DataFrame. Cleaned dataset.
    """
    categories = df['categories'].str.split(pat=';', expand=True)
    category_colnames = categories.iloc[0].str[:-2].values
    categories.columns = category_colnames
    categories = categories.applymap(lambda x: int(x[-1]))
    categories['related'] = categories['related'].replace(2, 0)
    
    columns_to_drop = ['categories', 'child_alone', 'original']
    for col in columns_to_drop:
        if col in df.columns:
            df.drop(col, axis=1, inplace=True)
    df = pd.concat([df, categories], axis=1)
    df.drop_duplicates(inplace=True)
    
    return df

def save_data(df, database_filename):
    """
    Save cleaned data to an SQLite database.
    
    Args:
    df: DataFrame. Cleaned dataset returned from clean_data().
    database_filename: string. Filename for the SQLite database.
    
    Returns:
    None
    """
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('DisasterResponse', engine, index=False, if_exists='replace')

def main():
    parser = argparse.ArgumentParser(description="ETL Pipeline to process and clean disaster response data")
    
    default_messages_path = '/Users/luca/Documents/Udacity - all learning materials/disaster_response_pipeline_project/data/disaster_messages.csv'
    default_categories_path = '/Users/luca/Documents/Udacity - all learning materials/disaster_response_pipeline_project/data/disaster_categories.csv'
    default_database_path = '/Users/luca/Documents/Udacity - all learning materials/disaster_response_pipeline_project/data/DisasterResponse.db'
    
    parser.add_argument("--data_messages", type=str, default=default_messages_path,
                        help=f"Filepath for the messages dataset. Default: {default_messages_path}")
    parser.add_argument("--data_categories", type=str, default=default_categories_path, 
                        help=f"Filepath for the categories dataset. Default: {default_categories_path}")
    parser.add_argument("--database", type=str, default=default_database_path, 
                        help=f"Filename for the SQLite database. Default: {default_database_path}")

    args = parser.parse_args()

    messages_filepath = args.data_messages
    categories_filepath = args.data_categories
    database_filepath = args.database

    print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
          .format(messages_filepath, categories_filepath))
    df = load_data(messages_filepath, categories_filepath)

    print('\nCleaning data...')
    df = clean_data(df)

    print('\nSaving data...\n    DATABASE: {}'.format(database_filepath))
    save_data(df, database_filepath)

    print('\nCleaned data saved to database!')

if __name__ == '__main__':
    main()
