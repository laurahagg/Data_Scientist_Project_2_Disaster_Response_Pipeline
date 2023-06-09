# Disaster Response Pipeline Project

### Project Overview:
In this project i built a machine learning pipeline to classify messages during disasters.
The Messages are spilt into 36 categories. 
First i cleaned up the data and saved it into a database and in the second step i worte a machine learning pipeline to classify this messages.

### Structure of this project:

This are the folders and the files of the project:

- app:
template, 
master.html (main page of web app),
go.html (classification result page of web app),
run.py (Flask file that runs app)

- data
disaster_categories.csv  and 
disaster_messages.csv  (data to process),
process_data.py,
ResponseData.db   (database to save and clean data),
ETL_Pipeline_Preparation.ipynb (juypter notebook with the data cleaning process)

- models
train_classifier.py,
classifier.pkl  (saved model),
ML_Pipeline_Preparation.ipynb (jupyter notebook with the ML algorithm)

- README.md

### Installation:
This project requires Python 3.11 and the following libraries:

- Data Science: Numpy and Pandas
- Machine Learning: SkicitLerarn, Pickle
- Natural Language Process: NLTK
- SQLlite Database: SQLalchemy
- Web App: Flask
- Data Visualisation: Plotly

To clone the GitHub Repository:
git@github.com:laurahagg/Data_Scientist_Project_2_Disaster_Response_Pipeline.git


### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/


### Acknowledgments:
This Project is part of the Nanodegree "Data Scientist" from the Udacity Codingschool.



