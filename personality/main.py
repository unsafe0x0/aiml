import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.impute import SimpleImputer

# Load dataset
data = pd.read_csv('dataset.csv')

# Preprocess data
X = data.drop('Personality', axis=1)
X = pd.get_dummies(X, drop_first=True)
y = data['Personality']

# Standardize features
scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# Handle missing values
imputer = SimpleImputer(strategy='mean')
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluate model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Custom input for prediction
custom_input = pd.DataFrame([{
"Time_spent_Alone": 7,
"Stage_fear": 1,
"Social_event_attendance": 1,
"Going_outside": 1,
"Drained_after_socializing": 5,
"Friends_circle_size": 1,
"Post_frequency": 2
}])

custom_input = pd.get_dummies(custom_input)
custom_input = custom_input.reindex(columns=X.columns, fill_value=0)

custom_input = pd.DataFrame(imputer.transform(custom_input), columns=X.columns)
custom_input = pd.DataFrame(scaler.transform(custom_input), columns=X.columns)

prediction = model.predict(custom_input)

print("Predicted Personality:", prediction[0])

# Results
# Accuracy: 0.9258620689655173
# Confusion Matrix:
#  [[277  25]
#  [ 18 260]]
# Classification Report:
#                precision    recall  f1-score   support

#    Extrovert       0.94      0.92      0.93       302
#    Introvert       0.91      0.94      0.92       278

#     accuracy                           0.93       580
#    macro avg       0.93      0.93      0.93       580
# weighted avg       0.93      0.93      0.93       580

# Predicted Personality: Introvert