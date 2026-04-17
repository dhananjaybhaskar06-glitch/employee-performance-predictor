import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# ---------------------------
# 1. CREATE DATA
# ---------------------------
np.random.seed(42)

data = pd.DataFrame({
    'Experience': np.random.randint(1, 20, 200),
    'Salary': np.random.randint(20000, 100000, 200),
    'Training_Hours': np.random.randint(10, 100, 200),
})

# Target
data['Performance'] = (
    data['Experience'] * 0.3 +
    data['Training_Hours'] * 0.2 +
    data['Salary'] * 0.0001 > 15
).astype(int)

# Save dataset
data.to_csv("data/employees.csv", index=False)

# ---------------------------
# 2. SPLIT
# ---------------------------
X = data[['Experience', 'Salary', 'Training_Hours']]
y = data['Performance']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# ---------------------------
# 3. MODEL
# ---------------------------
model = RandomForestClassifier()
model.fit(X_train, y_train)

# ---------------------------
# 4. PREDICT
# ---------------------------
y_pred = model.predict(X_test)

# ---------------------------
# 5. EVALUATION
# ---------------------------
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

cm = confusion_matrix(y_test, y_pred)

# Save confusion matrix
sns.heatmap(cm, annot=True, fmt='d')
plt.title("Confusion Matrix")
plt.savefig("images/confusion_matrix.png")
plt.clf()

# ---------------------------
# 6. VISUALIZATION
# ---------------------------
sns.scatterplot(x=data['Experience'], y=data['Salary'], hue=data['Performance'])
plt.title("Experience vs Salary")
plt.savefig("images/experience_vs_salary.png")
plt.clf()

print("✅ Files saved in images/ and data/")