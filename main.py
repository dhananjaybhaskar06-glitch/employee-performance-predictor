import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# ---------------------------
# 1. CREATE REALISTIC HR DATA
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

# ---------------------------
# 2. PERFORMANCE LOGIC
# ---------------------------
data['Performance'] = (
    (data['Experience'] * 0.3 +
     data['Training_Hours'] * 0.2 +
     data['Salary'] * 0.0001) > 18
).astype(int)

# Save dataset
data.to_csv("data/employees.csv", index=False)

# ---------------------------
# 3. ENCODING
# ---------------------------
le = LabelEncoder()
data['Department'] = le.fit_transform(data['Department'])
data['Education'] = le.fit_transform(data['Education'])

# ---------------------------
# 4. SPLIT
# ---------------------------
X = data.drop('Performance', axis=1)
y = data['Performance']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# ---------------------------
# 5. MODEL
# ---------------------------
model = RandomForestClassifier()
model.fit(X_train, y_train)

# ---------------------------
# 6. PREDICT
# ---------------------------
y_pred = model.predict(X_test)

# ---------------------------
# 7. EVALUATION
# ---------------------------
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

cm = confusion_matrix(y_test, y_pred)

sns.heatmap(cm, annot=True, fmt='d')
plt.title("Confusion Matrix")
plt.savefig("images/confusion_matrix.png")
plt.clf()

# ---------------------------
# 8. MORE GRAPHS (IMPORTANT)
# ---------------------------
sns.boxplot(x=data['Department'], y=data['Salary'])
plt.title("Salary by Department")
plt.savefig("images/salary_by_department.png")
plt.clf()

sns.boxplot(x=data['Education'], y=data['Performance'])
plt.title("Performance by Education")
plt.savefig("images/performance_by_education.png")
plt.clf()

print("✅ Project upgraded successfully!")