# German Credit Risk Classification

## Overview
This project analyzes and classifies credit risk using the German Credit dataset. The dataset is obtained from OpenML and contains information about individuals applying for credit, labeled as either "good" or "bad" credit risks. The project applies various machine learning models to predict credit risk based on given features.

## Project Workflow
The project consists of three main steps:

1. **Load and Explore Data**:
   - The dataset is fetched from OpenML.
   - The structure of the dataset is examined.
   - A visualization of class distribution is generated.

2. **Preprocess Data**:
   - Missing values are handled using appropriate imputation strategies.
   - Categorical features are encoded using OneHotEncoding.
   - Numerical features are standardized using StandardScaler.
   - The dataset is split into training and test sets.

3. **Train and Evaluate Models**:
   - Three machine learning models are trained: Logistic Regression, Decision Tree, and Random Forest.
   - Model performance is evaluated using accuracy and classification reports.
   - Confusion matrices are visualized to assess model predictions.

## Installation & Requirements
To run this project, ensure you have the following dependencies installed:

```sh
pip install pandas seaborn matplotlib scikit-learn openml
```

## How to Run the Project
Run the script using:

```sh
python script_name.py
```

## Visualizations
The project generates the following visualizations:
- A bar chart showing the distribution of credit risk (good vs. bad)
- Confusion matrices for each trained model to assess performance

## Key Functions

- `load_and_explore_data()`: Loads and visualizes the dataset.
- `preprocess_data(df)`: Preprocesses data by handling missing values, encoding categorical variables, and scaling numerical features.
- `train_and_evaluate_models(X_train, X_test, y_train, y_test)`: Trains models and evaluates them with classification reports and confusion matrices.

## Models Used
The following machine learning models are trained and evaluated:
1. Logistic Regression
2. Decision Tree Classifier
3. Random Forest Classifier

## Expected Output
For each model, the script prints:
- Accuracy score
- Classification report
- Confusion matrix visualization

## Author
Erdal Beyoglu

## License
This project is open-source and available under the [MIT License](LICENSE).
