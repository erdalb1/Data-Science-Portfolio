# German Credit Risk Prediction

## Project Overview
The goal of this project is to predict whether a customer is likely to default on a loan (bad credit) or not (good credit) based on various financial and demographic factors. We use the German Credit Dataset from OpenML and apply machine learning algorithms to build a predictive model.

## Objective
To build a classification model that predicts whether a customer has good or bad credit using various features such as account type, credit history, age, employment, and other financial factors.

## Tools & Libraries Used
- **pandas** – for data manipulation
- **seaborn** & **matplotlib** – for data visualization
- **scikit-learn** – for machine learning
- **openml** – to fetch the dataset
- **train_test_split** & **StandardScaler** – for data preprocessing
- **Logistic Regression**, **Decision Trees**, **Random Forest** – for classification models

## Steps Involved

### 1. Load & Explore the Dataset
The dataset is loaded from OpenML. We first display the first few rows to understand the data structure and check the class distribution (good vs. bad credit).

### 2. Data Preprocessing
Data preprocessing steps include:
- Handling missing values with imputation.
- Encoding categorical variables using one-hot encoding.
- Normalizing numerical features with standard scaling.
- Splitting the dataset into training and testing sets.

### 3. Train Machine Learning Models
Three machine learning models are trained:
- **Logistic Regression**
- **Decision Tree Classifier**
- **Random Forest Classifier**

### 4. Model Evaluation
The models are evaluated based on accuracy and detailed classification reports (precision, recall, F1-score). The Random Forest model is currently providing the best results.

## How to Run the Project

### Prerequisites
Ensure you have the required Python libraries installed. You can install them using the following command:

```bash
pip install pandas seaborn matplotlib scikit-learn openml
