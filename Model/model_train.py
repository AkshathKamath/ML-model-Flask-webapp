import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
import joblib



## Load df
df = pd.read_csv("../Datasets/train.csv")

## Categorising into numerical & categorical features
numerical_cols = df.select_dtypes(include=['int64','float64']).columns.tolist()
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
categorical_cols.remove('Loan_Status')
categorical_cols.remove('Loan_ID')

## Mode Imputation of Categorical cols
for col in categorical_cols:
    df[col].fillna(df[col].mode()[0], inplace=True)

## Median Imputation of Numerical cols
for col in numerical_cols:
    df[col].fillna(df[col].median(), inplace=True)

## Outliers Handling
df[numerical_cols] = df[numerical_cols].apply(lambda x: x.clip(*x.quantile([0.05, 0.95])))

## Log Transforamtion & Adding Income cols to single col
df['LoanAmount'] = np.log(df['LoanAmount']).copy()
df['TotalIncome'] = df['ApplicantIncome'] + df['CoapplicantIncome']
df['TotalIncome'] = np.log(df['TotalIncome']).copy()


## Dropping ApplicantIncome and CoapplicantIncome cols
df = df.drop(columns=['ApplicantIncome','CoapplicantIncome'])

## Label encoding categorical features
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

## Encode target column
df['Loan_Status'] = le.fit_transform(df['Loan_Status'])

## Split into independent & target features
X = df.drop(columns=['Loan_Status', 'Loan_ID'])
y = df.Loan_Status
RANDOM_SEED = 6

## Train-test split not needed as we build model on entire dataset and predict with values provided in input

## RandomForest
rf = RandomForestClassifier(random_state=RANDOM_SEED)
param_grid_forest = {
    'n_estimators': [200,400, 700],
    'max_depth': [10,20,30],
    'criterion' : ["gini", "entropy"],
    'max_leaf_nodes': [50, 100]
}
## Hyperparameter Tuning
grid_forest = GridSearchCV(
        estimator=rf,
        param_grid=param_grid_forest, 
        cv=5, 
        n_jobs=-1, 
        scoring='accuracy',
        verbose=0
    )
## Creating the model
model_forest = grid_forest.fit(X, y)

## Saving the model
joblib.dump(model_forest, './Trained-Models/RF_loan_model.joblib')

# ## Loading the model
# loaded_model = joblib.load('./Trained-Models/RF_loan_model.joblib')

# data = [[
#                 1.0,
#                 0.0,
#                 0.0,
#                 0.0,
#                 0.0,
#                 4.98745,
#                 360.0,
#                 1.0,
#                 2.0,
#                 8.698
#             ]]
# print(f"Prediction is : {loaded_model.predict(pd.DataFrame(data))}") ## Generate prediction to test