import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# ---------------------------
# 1. CREATE DATA
# ---------------------------
np.random.seed(42)

n = 300

data = pd.DataFrame({
    'Age': np.random.randint(22, 60, n),
    'Experience': np.random.randint(1, 20, n),
    'Salary': np.random.randint(20000, 150000, n),
    'Training_Hours': np.random.randint(10, 100, n),
    'Department': np.random.choice(['HR', 'IT', 'Sales', 'Finance'], n),
    'Education': np.random.choice(['Bachelors', 'Masters', 'PhD'], n),
})

# Target
data['Performance'] = (
    (data['Experience'] * 0.3 +
     data['Training_Hours'] * 0.2 +
     data['Salary'] * 0.0001) > 18
).astype(int)

# Save dataset
data.to_csv("data/employees.csv", index=False)

# ---------------------------
# 2. ENCODING
# ---------------------------
le_dept = LabelEncoder()
le_edu = LabelEncoder()

data['Department'] = le_dept.fit_transform(data['Department'])
data['Education'] = le_edu.fit_transform(data['Education'])

# ---------------------------
# 3. SPLIT
# ---------------------------
X = data.drop('Performance', axis=1)
y = data['Performance']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# ---------------------------
# 4. MODEL
# ---------------------------
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "models/performance_model.pkl")

# ---------------------------
# 5. PREDICT
# ---------------------------
y_pred = model.predict(X_test)

# ---------------------------
# 6. EVALUATION
# ---------------------------
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

cm = confusion_matrix(y_test, y_pred)

sns.heatmap(cm, annot=True, fmt='d')
plt.title("Confusion Matrix")
plt.savefig("images/confusion_matrix.png")
plt.clf()

# ---------------------------
# 7. FEATURE IMPORTANCE
# ---------------------------
importance = model.feature_importances_
features = X.columns

plt.barh(features, importance)
plt.title("Feature Importance")
plt.savefig("images/feature_importance.png")
plt.clf()

# ---------------------------
# 8. NEW EMPLOYEE PREDICTION
# ---------------------------
new_employee = pd.DataFrame({
    'Age': [30],
    'Experience': [5],
    'Salary': [50000],
    'Training_Hours': [40],
    'Department': [le_dept.transform(['IT'])[0]],
    'Education': [le_edu.transform(['Masters'])[0]]
})

prediction = model.predict(new_employee)

print("New Employee Prediction (1=High, 0=Low):", prediction[0])

print("✅ Advanced project completed!")