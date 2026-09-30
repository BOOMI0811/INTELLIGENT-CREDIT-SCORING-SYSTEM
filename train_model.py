import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

# Creating Sample Dataset (Loan data simulation)
np.random.seed(42)
data = {
    'Age': np.random.randint(21, 60, 500),
    'Income': np.random.randint(20000, 100000, 500),
    'Credit_Score': np.random.randint(300, 850, 500),
    'Loan_Amount': np.random.randint(50000, 500000, 500),
    'Default': np.random.choice([0, 1], 500, p=[0.8, 0.2]) # 0=No Default, 1=Default
}
df = pd.DataFrame(data)

X = df.drop('Default', axis=1)
y = df['Default']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")
print(f"F1-Score: {f1_score(y_test, y_pred):.2f}")
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Save model
import pickle
pickle.dump(model, open('credit_model.pkl','wb'))
print("\nModel Saved as credit_model.pkl")
