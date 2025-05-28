import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score, classification_report

# Step 1: Create a synthetic binary dataset
data = {
    "has_offer":      [1, 0, 1, 1, 0, 0, 1, 1, 0, 0],
    "is_weekend":     [0, 1, 1, 0, 0, 1, 0, 1, 0, 1],
    "user_active":    [1, 1, 0, 0, 1, 0, 1, 0, 1, 0],
    "click":          [1, 0, 0, 1, 0, 0, 1, 0, 0, 0]
}

df = pd.DataFrame(data)

# Step 2: Split features and target
X = df[["has_offer", "is_weekend", "user_active"]]
y = df["click"]

# Step 3: Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 4: Train Bernoulli Naive Bayes model
model = BernoulliNB()
model.fit(X_train, y_train)

# Step 5: Predict and evaluate
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
