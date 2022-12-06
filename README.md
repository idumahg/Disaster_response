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

#### 1. ETL Pipeline: 

  This first component is implemented in `process_data.py` and it:
  - Loads the `messages` and `categories` datasets.
  - Merges the two datasets.
  - Cleans the data.
  - Stores it in a SQLite database.

#### 2. ML Pipeline:

This component involves writing a machine learning pipeline that:
  - Loads data from the SQLite database.
  - Splits the dataset into training and testing sets.
  - Builds a text processing and machine learning pipeline
  - Trains and tunes a model using GridSearchCV.
  - Outputs the results on the test set.
  - Exports the final model as a pickle file.
  
This is given in the `train_classifier.py`.

#### 3. Flask Web App:

Here I use my knowledge of flask, html, css and javascript to build the web app. I also add to the web appdata visulaizations using Plotly.

The goal of this project is to build a "meme generator" – a multimedia application to dynamically generate memes, including an image with an overlaid quote. 

The quotes come from a variety of file types (PDF, TXT, DOCX and TXT). Hence we engineer a solution to extract each quote line-by-line from these files and save the images. 

There are sample quotes and images of Xander the pup in the data folder. However, the app also accepts dynamic user input through a command-line tool and a web service. 

With the flask web service, users can provide any image, together with a quote (with a body and author), and get an image with the quote inscribed onto the image.

## Instruction for setting up
* Install the dependencies by running 
  ```
  pip install -r requirements.txt
  ```
* To start the flask server run the following in terminal:
  ```
  python3 app.py
  ```
  This will provide a link where the server is located. The link will lead you to web service, where you caneither decide to generate a random image or create an image. To create an image, you will need to provide the image url, a quote and an author. 
* The images will be saved in the static folder.
* You can also get a random image by running the *meme.py* code in command line as described below.

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
