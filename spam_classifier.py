import pandas as pd

df = pd.read_csv('spam.csv', encoding='latin-1')

df = df[['v1', 'v2']]
df.columns = ['label', 'message']

print(df.head())
print(df.shape)
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

df['label'] = df['label'].map({'ham': 0, 'spam': 1})

X_train, X_test, y_train, y_test = train_test_split(
    df['message'], df['label'], test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

predictions = model.predict(X_test_tfidf)
print(f"✅ Accuracy: {accuracy_score(y_test, predictions) * 100:.2f}%")

def predict_message(msg):
    msg_tfidf = vectorizer.transform([msg])
    result = model.predict(msg_tfidf)
    return "🚨 SPAM" if result[0] == 1 else "✅ HAM (Not Spam)"

print(predict_message("Congratulations! You've won $1000! Claim now!"))
print(predict_message("FREE entry to win CASH PRIZE!! Text now!!"))
print(predict_message("Hey, can you send me the notes from today's class?"))
print(predict_message("Are you coming to the party tonight?"))
