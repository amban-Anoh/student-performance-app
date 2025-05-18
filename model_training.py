import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Load dataset
file_path = r"C:\Users\amban\ANA680\Week2\StudentsPerformance.csv"
df = pd.read_csv(file_path)

# Light EDA
print(df.info())
print(df['race/ethnicity'].value_counts())
print(df[['math score', 'reading score', 'writing score']].describe())

sns.pairplot(df, vars=['math score', 'reading score', 'writing score'], hue='race/ethnicity')
plt.suptitle("Score Distribution by Race/Ethnicity", y=1.02)
plt.tight_layout()
plt.show()

# Prepare features and target
X = df[['math score', 'reading score', 'writing score']]
y = df['race/ethnicity']

# Encode target
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Train models
models = {
    "Random Forest": RandomForestClassifier(),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "KNN": KNeighborsClassifier(),
    "SVM": SVC()
}

best_model = None
best_score = 0
best_model_name = ""

print("\n📊 Model Accuracy Scores:")
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"{name}: {acc:.4f}")

    if acc > best_score:
        best_model = model
        best_score = acc
        best_model_name = name

# Save model and encoder
with open("model.pkl", "wb") as f:
    pickle.dump(best_model, f)

with open("label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)

print(f"\n Best model saved: {best_model_name} with accuracy {best_score:.4f}")
