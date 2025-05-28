from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# train data
X = [[2, 1], [4, 0], [2, 1], [4, 0], [2, 0]]  # [legs, can fly]
y = ['Bird', 'Mammal', 'Bird', 'Mammal', 'Mammal']

#split data
X_train,X_test,Y_train,Y_test = train_test_split (X,y, test_size=0.6, random_state=42)

# train model
model = MultinomialNB()
model.fit(X, y)

# test data
test = [[2, 1], [4, 0], [2, 0]]
predictions = model.predict(test)
print(predictions)
print("Accuracy:", accuracy_score(Y_test, predictions))



