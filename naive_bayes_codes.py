import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Step 1: Load Dataset
# Make sure 'spam.csv' is in the same folder
df = pd.read_csv('/home/stemland/Downloads/spam.csv', encoding='latin-1')[['v1', 'v2']]
df.columns = ['label', 'message']

# Step 2: Encode Labels (ham -> 0, spam -> 1)
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Step 3: Split Data
X_train, X_test, y_train, y_test = train_test_split(df['message'], df['label'], test_size=0.2, random_state=42)

# Step 4: Convert Text to Numbers
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Step 5: Train Naive Bayes Model
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Step 6: Predict on Test Set
y_pred = model.predict(X_test_vec)

# Step 7: Evaluate the Model
print(" Accuracy:", accuracy_score(y_test, y_pred))
print("\n Classification Report:\n", classification_report(y_test, y_pred))
print("\n Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
