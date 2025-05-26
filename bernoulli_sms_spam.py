import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Load SMS dataset
df = pd.read_csv("/home/stemland/sms_data/spam_detection/SMSSpamCollection.csv", sep='\t', header=None, names=['label', 'message'])

# Convert labels to binary
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Feature extraction
vectorizer = CountVectorizer(binary=True)
X = vectorizer.fit_transform(df['message'])
y = df['label']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train and predict
model = BernoulliNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluate
print ("UCI SMS Spam Collection Dataset Output")
print("\n Accuracy:", accuracy_score(y_test, y_pred))
print("\n Classification Report:\n", classification_report(y_test, y_pred))

+333333