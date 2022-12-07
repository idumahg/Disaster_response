# Disaster Response Pipeline Project

This project contains codes written to analyze disaster data from [Appen](https://appen.com) to build a model for an API that classifies disaster messages.

The project also includes a web app where an emergency worker can input a new message and get classification results in several categories. This is a multi-output classification task.


## Table of Contents
* [Installation](#installation)
* [Project Motivation](#project-motivation)
* [Overview of project](#overview-of-project)
* [Instruction for setting up](#instruction-for-setting-up)
* [Results](#results)
* [Acknowledgements](#acknowledgements)


## Installation

There is no major libraries required to run the code beyond what is provided in the Anaconda python distribution. The code can be run with any version of Python 3.

## Project Motivation

Following a disaster, disaster response organizations get millions of communications either direct or via social media at the time when they have the least capacity to filter and pull out the messages which are most important. 

The way disaster is responded to is that different organizations will take care of different part of the problems. One might be in charge of water, blocked roads, fire etc. However, it is usually the case that there is only one in a thousand messages that might be relevant to the disaster response professionals. 

Therefore supervised machine learning based approaches are used and are more accurate than key word searches to analyze the data and know which of the organizations should respond to which need.

## Overview of project

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


## Results

Here I provide a snapshot of the built web app.

![Screen Shot 2022-12-06 at 6 35 05 PM](https://user-images.githubusercontent.com/55643305/206059673-569892b5-e43a-4b09-9c11-e95c42b8806f.png)
![Screen Shot 2022-12-06 at 7 58 12 PM](https://user-images.githubusercontent.com/55643305/206061741-25503eb8-11b1-4ebc-802d-e9300942978f.png)
![Screen Shot 2022-12-06 at 7 58 36 PM](https://user-images.githubusercontent.com/55643305/206061756-bf15e17e-8cca-4cf3-8b34-1028a1bd2adf.png)


## Acknowledgements
- This project was inspired by the [Data Science nanodegree program at Udacity](https://www.udacity.com/course/data-scientist-nanodegree--nd025).
- The data was provided for by [Appen](https://appen.com).
