# Weather dataset
data = {
    'Temperature': [30, 25, 27, 20, 32, 24, 33],
    'Humidity': [85, 80, 90, 70, 95, 65, 75],
    'Play': [0, 1, 0, 1, 0, 1, 1]  # 1: Yes, 0: No
}
df = pd.DataFrame(data)

X = df[['Temperature', 'Humidity']]
y = df['Play']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = GaussianNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print ("Weather Prediction Dataset Output")
print("\n Accuracy:", accuracy_score(y_test, y_pred))


