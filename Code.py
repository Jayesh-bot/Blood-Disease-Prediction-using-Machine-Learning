# 1. Import libraries

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# 2. Load dataset

df = pd.read_csv(r"D:\blood.csv")

# 3. Basic understanding (EDA)

print("Dataset Shape:", df.shape)
print("\nColumns:", df.columns.tolist())
print("\nFirst 5 rows:\n", df.head())

# 4. Visualization

target = df.columns[-1]   # last column

# Target distribution
plt.figure()
sns.countplot(x=df[target])
plt.title("Target Distribution")
plt.xticks(rotation=45)
plt.show()

# Correlation heatmap
plt.figure(figsize=(8,6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()


# 5. Prepare data

X = df.drop(columns=[target])
y = df[target]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# 6. Train model

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)


# 7. Evaluate model

pred = model.predict(X_test)

# Accuracy
print("\nAccuracy:", accuracy_score(y_test, pred))

# Precision, Recall, F1-score
print("\nClassification Report:\n")
print(classification_report(y_test, pred))

# Confusion Matrix
cm = confusion_matrix(y_test, pred)

plt.figure()
sns.heatmap(cm, annot=True, fmt="d")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# 8. Feature Importance

importance = pd.Series(model.feature_importances_, index=X.columns)
importance.sort_values().plot(kind='barh')
plt.title("Feature Importance")
plt.show()

# 9. Prediction
sample = X_test[0:1]
print("\nPrediction:", model.predict(sample))
print("Actual:", y_test.iloc[0])
