# Disaster Response Pipeline Project

### Project Overview:
In this project i built a machine learning pipeline to classify messages during disasters.
The Messages are spilt into 36 categories. 
First i cleaned up the data and saved it into a database and in the second step i worte a machine learning pipeline to classify this messages.

### Structure of this project:

This are the folders and the files of the project:

- app
| -- template
| |--- master.html  # main page of web app
| |--- go.html  # classification result page of web app
|-- run.py  # Flask file that runs app

- data
|- disaster_categories.csv  # data to process 
|- disaster_messages.csv  # data to process
|- process_data.py
|- InsertDatabaseName.db   # database to save clean data to
|- ETL_Pipeline_Preparation.ipynb #juypter notebook with the data cleaning process

- models
|- train_classifier.py
|- classifier.pkl  # saved model 
|- ML_Pipeline_Preparation.ipynb #jupyter notebook with the ML algorithm

- README.md

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/


