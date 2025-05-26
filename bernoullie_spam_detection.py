from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# Features: [buy, free, win, click, hello, congratulations]
X = [
    [1, 1, 1, 0, 0, 1],  # spam
    [0, 0, 0, 0, 1, 0],  # ham
    [1, 1, 0, 1, 0, 0],  # spam
    [0, 0, 0, 0, 1, 0],  # ham
    [0, 0, 0, 1, 0, 1],  # spam
    [0, 0, 0, 0, 1, 0],  # ham
]

y = [1, 0, 1, 0, 1, 0]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create and train model
model = BernoulliNB()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
print ("Small Spam Detection Dataset Output")
print("\n Predicted:", y_pred)
print("\n Actual:   ", y_test)
print("\n Accuracy:", accuracy_score(y_test, y_pred))
print("\n Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\n Classification Report:\n", classification_report(y_test, y_pred))


