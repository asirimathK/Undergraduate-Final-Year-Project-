import os
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, flash
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

app = Flask(__name__)
app.secret_key = 'supersecretkey'


# Train the Naive Bayes model
def train_naive_bayes():
    data = pd.read_csv("fypDS.csv")
    X = data['text']
    y = data['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    # Vectorization and training
    vectorizer = TfidfVectorizer()
    X_train_tfidf = vectorizer.fit_transform(X_train)

    model = MultinomialNB()
    model.fit(X_train_tfidf, y_train)

    return model, vectorizer


# Load and train the model
model, vectorizer = train_naive_bayes()


# Home page route
@app.route('/')
def index():
    return render_template('index.html')


# Handle form submission
@app.route('/submit', methods=['POST'])
def submit():
    if request.method == 'POST':
        user_comment = request.form['comment']

        if not user_comment:
            flash('Please enter a comment!', 'error')
            return redirect(url_for('index'))

        # Preprocess and validate the user input
        user_comment_tfidf = vectorizer.transform([user_comment])
        prediction = model.predict(user_comment_tfidf)
        probabilities = model.predict_proba(user_comment_tfidf)[0]

        if prediction[0] == 1:  # Assuming '1' means hate speech
            flash(f"Warning: This comment is classified as hate speech.", 'danger')
        else:
            # Save the non-hate speech comment
            with open('non_hate_speech.txt', 'a') as f:
                f.write(user_comment + '\n')
            flash(f"Success: This comment is not classified as hate speech.", 'success')

    return redirect(url_for('index'))


# Run the app
if __name__ == "__main__":
    app.run(debug=True)
