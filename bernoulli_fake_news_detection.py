from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score


# Sample data
texts = ["Breaking news: the economy is crashing", "This is fake news", "Scientists discover cure", "You won a prize", "Government confirms report"]
labels = [0, 1, 0, 1, 0]  # 1 = fake, 0 = real

# Vectorize using binary features
vectorizer = CountVectorizer(binary=True)
X = vectorizer.fit_transform(texts)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.3, random_state=42)

# Train and evaluate
model = BernoulliNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print ("Fake news Output")
print("\n Accuracy:", accuracy_score(y_test, y_pred))











