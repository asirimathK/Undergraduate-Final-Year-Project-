from flask import Flask, render_template, request, jsonify
import pickle

app = Flask(__name__)

# Load the pre-trained model and vectorizer
model = pickle.load(open("models/random_forest_model.pkl", "rb"))
vectorizer = pickle.load(open("models/tfidf_vectorizer.pkl", "rb"))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/submit', methods=['POST'])
def submit_comment():
    data = request.json  # Get the JSON data from the POST request
    user_comment = data.get('comment', '')

    # Preprocess the comment and predict
    user_comment_tfidf = vectorizer.transform([user_comment])
    prediction = model.predict(user_comment_tfidf)

    if prediction[0] == 1:  # Assuming '1' means hate speech
        return jsonify({'status': 'blocked'})  # Return JSON indicating the comment is blocked
    else:
        return jsonify({'status': 'allowed'})  # Return JSON indicating the comment is allowed

if __name__ == '__main__':
    app.run(debug=True)
