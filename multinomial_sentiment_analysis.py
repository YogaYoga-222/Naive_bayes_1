from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Example data
texts = ["I loved the movie", "It was a terrible movie", "Fantastic acting", "Worst film ever", "Great story", "Not that much worth"]
labels = [1, 0, 1, 0, 1, 0]  # 1 = positive, 0 = negative

# Convert text to feature vectors
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.3, random_state=42)

# Train model
model = MultinomialNB()
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
print("Movie Review Sentiment Analysis Output")
print("\n Accuracy:", accuracy_score(y_test, y_pred))










