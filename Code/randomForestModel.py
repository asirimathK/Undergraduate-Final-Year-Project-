import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import os

# Function to train Random Forest model
def train_random_forest(X_train, y_train):
    # Fill missing values in the training data with empty strings
    X_train = X_train.fillna('')

    # Initialize TfidfVectorizer
    vectorizer = TfidfVectorizer()

    # Transform the training data
    X_train_tfidf = vectorizer.fit_transform(X_train)

    # Initialize Random Forest Classifier
    model = RandomForestClassifier(random_state=42)

    # Train the Random Forest model
    model.fit(X_train_tfidf, y_train)

    return model, vectorizer

# Function to predict and evaluate the model
def evaluate_model(model, vectorizer, X_test, y_test):
    # Fill missing values in the test data with empty strings
    X_test = X_test.fillna('')

    # Transform the test data
    X_test_tfidf = vectorizer.transform(X_test)

    # Predict the labels for the test data
    y_pred = model.predict(X_test_tfidf)

    # Evaluate the model
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

# Main logic
if __name__ == "__main__":
    # Load dataset (fypDS.csv contains both hate speech and non-hate speech entries)
    data = pd.read_csv("fypDS.csv")

    # Assuming 'text' is the feature column and 'label' is the target column
    X = data['text']
    y = data['label']

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

    # Train the Random Forest model
    model, vectorizer = train_random_forest(X_train, y_train)

    # Evaluate the model on the test data
    evaluate_model(model, vectorizer, X_test, y_test)

    while True:
        user_comment = input("Enter a comment to validate (or type 'exit' to quit): ")

        if user_comment.lower() == 'exit':
            print("Exiting the program.")
            break

        # Preprocess and validate the user input
        user_comment_tfidf = vectorizer.transform([user_comment])

        # Predict the label and calculate the probabilities
        prediction = model.predict(user_comment_tfidf)
        probabilities = model.predict_proba(user_comment_tfidf)[0]

        # Display the prediction and probabilities
        print(f"Probability of non-hate speech: {probabilities[0]:.4f}")
        print(f"Probability of hate speech: {probabilities[1]:.4f}")

        if prediction[0] == 1:  # Assuming '1' means hate speech
            print("This comment is classified as hate speech.")
            # Save the comment if it is flagged as hate speech
            df = pd.DataFrame({'statement': [user_comment]})
            if os.path.isfile("hateout.csv"):
                df.to_csv("hateout.csv", mode='a', header=False, index=False)
            else:
                df.to_csv("hateout.csv", mode='w', header=True, index=False)
            print(f"Flagged comment saved to hateout.csv")
        else:
            print("This comment is not classified as hate speech.")
