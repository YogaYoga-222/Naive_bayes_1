from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Sample movie review data
reviews = [
    "I love the movie",          # Positive
    "Fantastic acting",          # Positive
    "Great story",               # Positive
    "Amazing movie",             # Positive
    "I hated the film",          # Negative
    "Worst acting",              # Negative
    "Terrible plot",             # Negative
    "Bad movie",                 # Negative
]

labels = [1, 1, 1, 1, 0, 0, 0, 0]  # 1 = Positive, 0 = Negative

# Step 1: Convert text to numerical features
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(reviews)

# Step 2: Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.3, random_state=42)

# Step 3: Train the Multinomial Naive Bayes model
model = MultinomialNB()
model.fit(X_train, y_train)

# Step 4: Predict on test data
y_pred = model.predict(X_test)

# Step 5: Evaluate the model
print ("Movie Review Data Output")
print("\n Accuracy:", accuracy_score(y_test, y_pred))

# (Optional) Show predictions
print("\n Predicted labels:", y_pred)
print("\n Actual labels   :", y_test)

