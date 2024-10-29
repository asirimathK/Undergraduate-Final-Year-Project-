from flask import Flask, render_template, request, jsonify
import pickle
import pandas as pd
import os

app = Flask(__name__)

# Load the pre-trained model and vectorizer
model = pickle.load(open("models/random_forest_model.pkl", "rb"))
vectorizer = pickle.load(open("models/tfidf_vectorizer.pkl", "rb"))

# Load the offensive language dataset
offensive_data = pd.read_csv("offensive.csv")
offensive_texts = set(offensive_data['text'].str.lower())  # Convert to lowercase for case-insensitive matching


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/submit', methods=['POST'])
def submit_comment():
    data = request.json  # Get the JSON data from the POST request
    user_comment = data.get('comment', '').strip().lower()  # Convert to lowercase for matching

    if not user_comment:  # Check if the comment is empty
        return jsonify({'status': 'error', 'message': 'Please enter a comment.'})

    # Check if the comment is in the offensive dataset
    if user_comment in offensive_texts:
        # Record the offensive comment in hateout.csv
        df = pd.DataFrame({'comment': [user_comment]})

        # Check if 'hateout.csv' exists, append without header if it does, create new file otherwise
        if os.path.isfile("hateout.csv"):
            df.to_csv("hateout.csv", mode='a', header=False, index=False)
        else:
            df.to_csv("hateout.csv", mode='w', header=True, index=False)

        # Return JSON response indicating the comment is blocked as hate speech
        return jsonify({'status': 'blocked', 'message': 'Comment blocked because it contains offensive language.'})

    # Preprocess the comment and predict using the model if not in offensive dataset
    user_comment_tfidf = vectorizer.transform([user_comment])
    prediction = model.predict(user_comment_tfidf)

    if prediction[0] == 1:  # If the comment is predicted as hate speech
        # Record the hate speech comment in hateout.csv
        df = pd.DataFrame({'comment': [user_comment]})

        if os.path.isfile("hateout.csv"):
            df.to_csv("hateout.csv", mode='a', header=False, index=False)
        else:
            df.to_csv("hateout.csv", mode='w', header=True, index=False)

        # Return JSON response indicating the comment is blocked
        return jsonify({'status': 'blocked', 'message': 'Comment blocked because it contains hate speech.'})
    else:
        # Return JSON response indicating the comment is allowed
        return jsonify({'status': 'allowed', 'message': 'Comment allowed.'})


if __name__ == '__main__':
    app.run(debug=True)
