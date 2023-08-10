# Disaster Response Pipeline Project

## Table of Contents

1. [Project Overview](#overview)
2. [Project Components](#components)
    - [ETL Pipeline](#ETL)
    - [Machine Learning Pipeline](#ml)
    - [Flask Web App](#flask)
3. [Getting Started](#gettingstarted)
    - [Dependencies](#dependencies)
    - [Installation](#installation)
    - [Execution](#execution)
4. [File Descriptions](#files)
5. [Results](#results)
6. [Licensing, Authors, and Acknowledgements](#licensing)

## Project Overview <a name="overview"></a>

This project is part of the Data Science Nanodegree Program by Udacity in collaboration with Figure Eight. The initial dataset contains pre-labelled tweet and messages from real-life disasters. The aim of the project is to build a Natural Language Processing tool that categorize messages.

The Project is divided into the following Sections:

1. Data Processing, ETL Pipeline to extract data from source, clean data and save them in a proper database structure.
2. Machine Learning Pipeline to train a model able to classify text message in categories.
3. Web App to show model results in real time.

## Project Components <a name="components"></a>

There are three components of this project:

### 1. ETL Pipeline (Process Data) <a name="ETL"></a>

A Python script, `etl_pipeline.py`, is written to perform the following tasks:

- Load the `messages` and `categories` datasets
- Merge the two datasets
- Clean the data
- Store the data in a SQLite database

This script takes three command line arguments:

- `messages_filepath` - The file path of the messages csv file
- `categories_filepath` - The file path of the categories csv file
- `database_filepath` - The file path for the cleaned data database

Run the script with the following command:

`python etl_pipeline.py disaster_messages.csv disaster_categories.csv DisasterResponse.db`

### 2. Machine Learning Pipeline (Build Model) <a name="ml"></a>

In progress

### 3. Flask Web App (Run Web App) <a name="flask"></a>

In progress

## Getting Started <a name="gettingstarted"></a>

### Dependencies <a name="dependencies"></a>

Python 3.5+ (I used Python 3.7)
- Machine Learning Libraries: NumPy, SciPy, Pandas, Scikit-Learn
- Natural Language Process Libraries: NLTK
- SQLite Libraries: SQLalchemy
- Web App and Data Visualization: Flask, Plotly

### Installation <a name="installation"></a>

Clone this GIT repository:

`git clone https://github.com/YOUR-USERNAME/YOUR-REPOSITORY`

### Execution <a name="execution"></a>

1. Run the following command in the project's root directory to set up your database and model.

    `python etl_pipeline.py disaster_messages.csv disaster_categories.csv DisasterResponse.db`

2. [TBD - other commands related to ML pipeline and Flask Web App]

## File Descriptions <a name="files"></a>

The files structure is arranged as below:

- Root:  
  - README.md: a descriptive file for instructions on the project.
  - etl_pipeline.py: ETL pipeline scripts to clean, load, and save data in a database.
  - [TBD - other files]

## Results <a name="results"></a>

[TBD]

## Licensing, Authors, and Acknowledgements <a name="licensing"></a>

Must give credit to Figure Eight for the data. You can find the Licensing for the data and other descriptive information at the Figure Eight's official website. Otherwise, feel free to use the code here as you would like!