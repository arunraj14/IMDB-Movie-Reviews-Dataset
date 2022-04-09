# IMDB-Movie-Reviews-Dataset

### *IMDB dataset has 27K movie reviews for natural language processing or Text analytics.*


This is a dataset for binary sentiment classification containing imbalanced dataset. A set of 22,000 highly polar movie reviews for training and 5,000 for testing. So, we will predict the number of positive and negative reviews using classification.

Here is a short description on the purpose of each file in this repository.

### IMDB Sentiment Analysis.ipynb
This Notebook has all the necessary steps from data exploration, data preprocessing and data modelling with various algorithms.It also the evaluation metrics for the final result

### classifier.pkl
Final model has been serialized as a pickle file which can be used directly in the target environment

### flask_app.py
REST API is built for ML prediction in Flask & Flasgger

### Dockerfile
Customized docker file for the IMDB Sentimental Analysis to build image for containers

### requirements.txt
All dependencies needed for the container
