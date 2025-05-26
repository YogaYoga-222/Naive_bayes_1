from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Features: [Buy, Win, Hello, Free]
X = [
    [1, 1, 0, 1],  # spam
    [0, 0, 1, 0],  # ham
    [1, 1, 0, 1],  # spam
    [0, 0, 1, 0],  # ham
]

# Labels: 1 for spam, 0 for ham
y = [1, 0, 1, 0]

# Train-test split (just for demonstration)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

# Create and train model
model = BernoulliNB()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
print("Predicted:", y_pred)
print("Actual:   ", y_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
