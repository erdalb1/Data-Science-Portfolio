German Credit Risk Prediction
This project aims to predict whether a customer is likely to default on a loan based on their financial and demographic factors. Using the German Credit dataset, we explore different machine learning models to classify individuals as either having a "good" or "bad" credit risk. The dataset is analyzed, preprocessed, and used to train multiple classification models to predict loan default risk.

Objective
To predict whether a customer will default on a loan using various financial and demographic factors.

Tools & Libraries Used
pandas: For data manipulation and analysis.
seaborn & matplotlib: For data visualization.
scikit-learn: For building machine learning models.
train_test_split, StandardScaler: For data preprocessing.
Logistic Regression, Decision Trees, Random Forest: For classification models.

Project Workflow


Step1 : Load & Explore the Dataset
We begin by loading the German Credit dataset from OpenML and exploring the data for any patterns or inconsistencies. The initial exploration includes checking for class balance and visualizing the distribution of the target variable (good vs. bad credit).

python
Copy
Edit
import openml  # To fetch the dataset
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset from OpenML
dataset = openml.datasets.get_dataset(31)  # German Credit Dataset
df, _, _, _ = dataset.get_data()

# Display first few rows
print(df.head())

# Check class balance
sns.countplot(x='class', data=df)
plt.title("Class Distribution: Good vs. Bad Credit Risk")
plt.show()

Step2: Data Preprocessing
The next step is to preprocess the data, which involves handling missing values, encoding categorical features, and scaling numerical variables. After preprocessing, the dataset is split into training and testing sets.

python
Copy
Edit
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

# Separate features and target variable
X = df.drop(columns=['class'])
y = df['class'].apply(lambda x: 1 if x == 'bad' else 0)

# Identify categorical & numerical columns
categorical_features = X.select_dtypes(include=['object']).columns
numerical_features = X.select_dtypes(exclude=['object']).columns

# Create preprocessing pipeline
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numerical_transformer, numerical_features),
    ('cat', categorical_transformer, categorical_features)
])

# Split into training & test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Transform data
X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)

Step3: Train Machine Learning Models
Several machine learning models are trained to predict loan default risk, including Logistic Regression, Decision Trees, and Random Forest. The accuracy of each model is evaluated and compared.

python
Copy
Edit
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Train models
models = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(n_estimators=100)
}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{name} Accuracy: {accuracy:.2f}")
    print(classification_report(y_test, y_pred))
    
Step4: Results and Next Steps
The Random Forest model provides the highest accuracy in predicting credit risk. Moving forward, I am planning to:

Tune hyperparameters for better performance.
Perform feature selection to optimize the model.
Explore additional models and compare their results.


Next Steps
Create a new repository on GitHub.
Upload german_credit_risk.py (the script).
Write a detailed README.md explaining the project.

Dataset Source
The dataset used in this project is the German Credit Dataset available on OpenML.
