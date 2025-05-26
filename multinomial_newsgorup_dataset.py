from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Load text data
data = fetch_20newsgroups(subset='all', categories=['sci.space', 'rec.autos', 'comp.graphics'], shuffle=True, random_state=42)

# Convert text to feature vectors
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(data.data)
y = data.target

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train model
model = MultinomialNB()
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
print ("20 Newsgroups Dataset Output")
print("\n Accuracy:", accuracy_score(y_test, y_pred))
print("\n Classification Report:\n", classification_report(y_test, y_pred))