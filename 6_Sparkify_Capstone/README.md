# Sparkify Churn Prediction Project

Predict propensity to churn for users of a music streaming app

## Project Overview
Sparkify is a fictitious music streaming app, similar to Spotify. The dataset contains user interaction logs for actions performed by users. The full dataset is 12 GB. In this project, we work with a mini dataset(128 MB) locally and medium dataset(230 MB) on IBM Watson cluster. 

## Project Motivation
The aims of the project are:
- Research and investigate a real-world problem of interest
- Accurately apply specific data science algorithms and techniques
- Properly analyze and visualize data and results for validity
- Document and report the work done

## Requirements
Anaconda Python-3.x distribution with Pyspark-2.4, specifically
- Pandas
- Numpy
- Pyspark
- Seaborn 
- Matplotlib

The mini dataset can be trained locally, but the medium dataset was trained using IBM Watson.

##  Files
- **Sparkify** - Jupyter notebook with EDA and modeling locally on mini dataset
- **Sparkify Capstone** - Jupyter notebook with feature engineering and modeling on medium dataset on IBM Watson
- **blog_pics/** - Folder with graphs and pics used in blog post
*Data files are not included due to large size*

## Data Analysis
Analyzed the dataset for pattern in user behaviour and documented the insights in blog post.

## Modeling
Since the dataset is imbalanced, F1 score was used as the metric for model performance.
We tried 4 different models - Logistic Regression, Random Forest, Linear SVC, Gradient-Boosted Trees. 
The best model was Random Forest, which out-performed the baseline model by 27 percent points on validation set. The performance on validation set for the best model was:

``
Accuracy: 0.9
F-1 Score:0.8916530278232406
``


## Challenges faced:
- Highly imbalanced dataset
- Large size of dataset

## Report
The blog post detailing insights from this project can be found [here](https://medium.com/@sakchhi.sri/churn-prediction-using-spark-1d8f6bd4092d).
