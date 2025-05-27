# Naive Bayes Classifier Projects

![Python](https://img.shields.io/badge/python-3.10%2B-blue)


This repository contains various Python projects using different types of Naive Bayes algorithms to solve classification problems with both small and large datasets.

---

## What You’ll Learn

- Basics and applications of **Naive Bayes** algorithms
- Difference between **GaussianNB**, **MultinomialNB**, and **BernoulliNB**
- How to use Naive Bayes on different datasets: text, numeric, and binary
- Evaluate model performance using accuracy, confusion matrix, and classification report

---

## Project Structure

### General Examples
- `naive_bayes_codes.py`: Large dataset example for spam detection
- `nave_bayes_code_1.py`: Small basic example to understand Naive Bayes

### Gaussian Naive Bayes
- `gaussian_iris_classification.py`: Iris flower classification (numeric features)
- `gaussian_weather_classify.py`: Weather prediction using small numeric dataset
- `gaussian_breast_cancer.py`: Breast cancer prediction with small dataset
- `gaussian_animal_classify.py`: Simple animal classification using numeric features like number of legs and flying ability

### Multinomial Naive Bayes
- `multinomial_movie_review.py`: Sentiment analysis on movie reviews
- `multinomial_newsgroup_dataset.py`: News article classification using 20 newsgroups dataset
- `multinomial_sentiment_analysis.py`: Text classification with word counts

### Bernoulli Naive Bayes
- `bernoulli_fake_news_detection.py`: Fake news detection using binary features
- `bernoulli_spam_detection.py`: Spam email classification
- `bernoulli_sms_spam.py`: SMS spam classification using binary word presence

### Datasets
- `spam.csv`: Dataset for email spam detection
- `SMSSpamCollection.csv`: Dataset for SMS spam classification

---

## How to Run the Code

### Requirements

Install the required libraries before running any script:

```bash
pip install scikit-learn pandas matplotlib
```
> **Note:** Use `pip3` if `pip` doesn't work on your system.

## Run a script

> **Note:** Use `python` if `python3` doesn't work on your system.

```bash 
python3 naive_bayes_codes.py
```
```bash 
python3 naive_bayes_code_1.py
```
```bash
python3 gaussian_iris_classification.py
```
```bash
python3 gaussian_weather_classify.py
```
```bash
python3 gaussian_breast_cancer.py
```
```bash
python3 gaussian_animal_classify.py
```
```bash
python3 multinomial_sentiment_analysis.py
```
```bash
python3 multinomial_newsgroup_dataset.py
```
```bash
python3 multinomial_movie_review.py
```
```bash
python3 bernoulli_fake_news_detection.py
```
```bash
python3 bernoulli_spam_detecion.py
```
```bash
python3 bernoulli_sms_spam.py
```

## Note

* All files are organized based on the Naive Bayes type.

* You can explore how each classifier performs with different datasets.

* This repo is designed for learning and practicing machine learning basics.
