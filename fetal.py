# Step 1: Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


# Step 2: Load Dataset
df = pd.read_csv("fetal_health.csv")  # dataset

# Step 3: Features & Labels
X = df.drop("fetal_health", axis=1)
y = df["fetal_health"]

# Step 4: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 6: Model Training
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Step 7: Evaluation
y_pred = model.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

cm = np.array([[328, 4, 1],
               [14, 49, 1],
               [1, 1, 27]])

# Step 3: Define class labels
labels = ["Class 1", "Class 2", "Class 3"]

# Step 4: Plot heatmap
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=labels, yticklabels=labels)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix Heatmap")
plt.show()



# Assume y_test & y_pred & define hear.....
report = classification_report(y_test, y_pred)
matrix = confusion_matrix(y_test, y_pred)

# Save the file
with open("report.txt", "w") as f:
    f.write("Confusion Matrix:\n")
    f.write(str(matrix))
    f.write("\n\nClassification Report:\n")
    f.write(report)
