import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

data = {
    'email': [
        "Win money now!!!",
        "Limited offer just for you",
        "Hey, are we still meeting tomorrow?",
        "Your account has been credited",
        "Congratulations, you won a lottery!",
        "Let's study together for exams",
        "Get cheap loans instantly",
        "Important update about your results",
        "Click here to claim your prize",
        "Can you send me the notes?"
    ],
    'label': [1,1,0,0,1,0,1,0,1,0]  
}

df = pd.DataFrame(data)
X = df['email']
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)
model = MultinomialNB()
model.fit(X_train_vec, y_train)
y_pred = model.predict(X_test_vec)
print("Accuracy:", accuracy_score(y_test, y_pred))
while True:
    user_input = input("\nEnter an email message (or type 'exit'): ")
    if user_input.lower() == 'exit':
        break
    input_vec = vectorizer.transform([user_input])
    prediction = model.predict(input_vec)
    if prediction[0] == 1:
        print("This is a SPAM email!")
    else:
        print("This is NOT a spam email.")
