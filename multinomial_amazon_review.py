import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Step 1: Load the dataset
url = "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/train.csv"
df = pd.read_csv(url, header=None)
df.columns = ["Category", "Title", "Description"]

# Combine title and description
df["Text"] = df["Title"] + " " + df["Description"]

# Map categories
label_map = {1: "World", 2: "Sports", 3: "Business", 4: "Sci/Tech"}
df["Category"] = df["Category"].map(label_map)

# Step 2: Split data
X_train, X_test, y_train, y_test = train_test_split(df["Text"], df["Category"], test_size=0.2, random_state=42)

# Step 3: TF-IDF Vectorization
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Step 4: Train Multinomial Naive Bayes
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Step 5: Predict and evaluate
y_pred = model.predict(X_test_vec)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

