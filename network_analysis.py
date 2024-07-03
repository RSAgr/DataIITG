import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

train_data = pd.read_csv('Train_Data.csv')
test_data = pd.read_csv('Test_Data.csv')

categorical_columns = train_data.select_dtypes(include=['object']).columns

le = LabelEncoder()

for column in categorical_columns:
    train_data[column] = le.fit_transform(train_data[column].astype(str))
    test_data[column] = le.transform(test_data[column].astype(str))

print(train_data.head())
print(test_data.head())

for column in train_data.columns:
    if train_data[column].isnull().sum() > 0:
        train_data[column].fillna(train_data[column].mode()[0], inplace=True)
        test_data[column].fillna(test_data[column].mode()[0], inplace=True)

X = train_data.drop('attack', axis=1)
y = train_data['attack']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_val)

print(accuracy_score(y_val, y_pred))
print(classification_report(y_val, y_pred))

test_predictions = model.predict(test_data)

sample_submission = pd.read_csv('Sample_Submission.csv')
sample_submission['attack'] = test_predictions
sample_submission.to_csv('submission.csv', index=False)
