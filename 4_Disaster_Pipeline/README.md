# Disaster Response Pipeline Project

Predict response category for messages related to disasters using Multi-Output Classifier with Web Interface using Flask

## Getting Started

Clone the repository and install dependencies

### Prerequisites

Use the standard anaconda base environment 
```
pip install -r requirements.txt
```
- Plotly - ```conda install -c plotly plotly```

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://127.0.0.1:3001/

Repository contains 2 trained models:
- *classifier* - Trained by lemmatizing words with POS information. This takes too long to train.
- *classifier2* - Trained by lemmatizing without POS information.
Currently the POS tag lemmatization code is commented out and model dumps classifer2 on training. But predicitons are made on classifier final model.

## License

This project is licensed under the MIT License.
