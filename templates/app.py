import pickle
from flask import Flask, request, render_template

app = Flask(__name__)

# Initialize the model and vectorizer as global variables
model = None
vectorizer = None

# Load the trained Random Forest model and fitted TfidfVectorizer
def load_model():
    global model, vectorizer
    try:
        model = pickle.load(open("random_forest_model.pkl", "rb"))
        vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))
        print("Model and vectorizer loaded successfully.")
    except Exception as e:
        print(f"Error loading model or vectorizer: {e}")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/submit', methods=['POST'])
def submit_comment():
    # Ensure model and vectorizer are loaded
    if model is None or vectorizer is None:
        load_model()

    user_comment = request.form['comment']

    # Preprocess and validate the user input
    user_comment_tfidf = vectorizer.transform([user_comment])

    # Predict the label and calculate the probabilities
    prediction = model.predict(user_comment_tfidf)
    probabilities = model.predict_proba(user_comment_tfidf)[0]

    if prediction[0] == 1:  # Assuming '1' means hate speech
        message = "This comment is classified as hate speech."
    else:
        message = "This comment is not classified as hate speech."

    return render_template('result.html', message=message)

if __name__ == "__main__":
    # Load model and vectorizer on app start
    load_model()
    app.run(debug=True)
