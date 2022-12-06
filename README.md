# Disaster Response Pipeline Project

This project contains codes written to analyze disaster data from [Appen](https://appen.com) to build a model for an API that classifies disaster messages.

The project also includes a web app where an emergency worker can input a new message and get classification results in several categories. The web app display
visualizations of the data. This is a multi-output classification task.


## Table of Contents
* [Overview of project](#overview-of-project)
* [Instruction for setting up](#instruction-for-setting-up)
* [General info](#general-info)
* [Acknowledgements](#acknowledgements)


## Overview of project.

There are three main components of this project:

#### ETL Pipeline: 

  This first component is implemented in `process_data.py` and it:
  - Loads the `messages` and `categories` datasets.
  - Merges the two datasets.
  - Cleans the data.
  - Stores it in a SQLite database.

#### ML Pipeline:

This component involves writing a machine learning pipeline that:
  - Loads data from the SQLite database.
  - Splits the dataset into training and testing sets.
  - Builds a text processing and machine learning pipeline
  - Trains and tunes a model using GridSearchCV.
  - Outputs the results on the test set.
  - Exports the final model as a pickle file.
  
This is given in the `train_classifier.py`.

#### Flask Web App:

- Here I use my knowledge of flask, html, css and javascript to build the web app. 
- I also add to the web appdata visulaizations using Plotly.

## Instruction for setting up
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
    
        ```python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db```
        
    - To run ML pipeline that trains classifier and saves
    
        ```python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl```

2. Go to `app` directory: `cd app`

3. Run your web app: `python run.py`

4. Click on the web link to visualize the app.


## General info
* Quote Engine: This module is composed of classes that ingest different file types that contain quotes. It contains:
  * An abstract base class **IngestorInterface** that contains class method signature.
  * An **Ingestor** class that realize the **IngestorInterface** class and encapsulates all the ingestors to provide one interface to load any supported    file type.
  * A **QuoteModel** class to encapsulate the body and author of a quote.
  * Ingestor classes (DocxIngestor, TextIngestor, PDFIngestor, CSVIngestor) to ingest file types.
* *meme.py* is a simple CLI code that can be run from the terminal. It takes in three *optional* CLI arguuments:
  * \--body: a string quote body
  * \--author: a string quote author
  * \--path: an image path
  
  The script returns a path to a generated image. If any argument is not defined, a random selection is used.
* *app.py* is a flask server code. This code uses the Quote Engine Module and Meme Generator Modules to generate a random captioned image. It uses the *requests* package to fetch an image from a user submitted URL.
* The HTML templates files are given in *templates/*
* Example quotes are provided in   _./\_data/SimpleLines_ and  _./\_data/DogQuotes_

## Acknowledgements
- This project was inspired by the Intermediate Python course at Udacity
- This project was based on [this nanodegree program](https://www.udacity.com/course/intermediate-python-nanodegree--nd303).
- Many thanks to Udacity for this opportunity to learn.
