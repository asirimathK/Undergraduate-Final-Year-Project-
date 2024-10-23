import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import os

def train_random_forest(X_train, y_train):
    # Fill missing values in the training data with empty strings
    X_train = X_train.fillna('')
    vectorizer = TfidfVectorizer()
    X_train_tfidf = vectorizer.fit_transform(X_train)
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train_tfidf, y_train)

    return model, vectorizer

def evaluate_model(model, vectorizer, X_test, y_test):
    # Fill missing values in the test data with empty strings
    X_test = X_test.fillna('')
    # Transform the test data
    X_test_tfidf = vectorizer.transform(X_test)
    # Predict the labels for the test data
    y_pred = model.predict(X_test_tfidf)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))


if __name__ == "__main__":

    data = pd.read_csv("fypDS.csv")

    X = data['text']
    y = data['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    model, vectorizer = train_random_forest(X_train, y_train)

    evaluate_model(model, vectorizer, X_test, y_test)

    while True:
        user_comment = input("Enter a comment to validate (or type 'exit' to quit): ")

        if user_comment.lower() == 'exit':
            print("Exiting the program.")
            break

        user_comment_tfidf = vectorizer.transform([user_comment])

        prediction = model.predict(user_comment_tfidf)
        probabilities = model.predict_proba(user_comment_tfidf)[0]

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
