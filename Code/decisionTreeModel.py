import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report


# Load the dataset and preprocess it
def load_and_preprocess_data(csv_file):
    try:
        # Load the CSV file
        data = pd.read_csv(csv_file)

        # Ensure the CSV contains the necessary columns
        if not {'text', 'label'}.issubset(data.columns):
            print("Error: The CSV file must contain 'text' and 'label' columns.")
            return None, None

        # Preprocess text: Convert to lowercase
        data['text'] = data['text'].str.lower()

        # Split the data into features (X) and labels (y)
        X = data['text']
        y = data['label']  # Assuming 'label' contains 1 for hate speech, 0 for non-hate speech

        return X, y
    except FileNotFoundError:
        print("Error: CSV file not found. Please check the file path.")
        return None, None


# Train a Decision Tree model on the dataset
def train_decision_tree(X_train, y_train):
    # Vectorize the text using TF-IDF
    vectorizer = TfidfVectorizer()
    X_train_tfidf = vectorizer.fit_transform(X_train)

    # Train the Decision Tree classifier
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train_tfidf, y_train)

    return model, vectorizer


# Function to validate user input and classify it using the trained model
def validate_user_input(model, vectorizer, user_input, output_file):
    # Preprocess the user input
    user_input = user_input.lower()

    # Vectorize the user input
    user_input_tfidf = vectorizer.transform([user_input])

    # Predict using the trained model
    prediction = model.predict(user_input_tfidf)

    # If the prediction is 1 (hate speech), save the comment in the output file
    if prediction == 1:
        print("The comment is classified as hate speech.")
        save_flagged_comment(output_file, user_input)
    else:
        print("The comment is classified as non-hate speech.")


# Function to save the flagged comment in the output CSV file
def save_flagged_comment(output_file, user_comment):
    # Create a DataFrame to store the flagged comment
    df = pd.DataFrame({'statement': [user_comment]})

    # Check if the output CSV file exists
    if os.path.isfile(output_file):
        # Append to the existing file
        df.to_csv(output_file, mode='a', header=False, index=False)
    else:
        # Create a new file with the header
        df.to_csv(output_file, mode='w', header=True, index=False)

    print(f"Flagged comment saved to {output_file}")


# Main logic
if __name__ == "__main__":
    # Load the dataset
    csv_file = "fypDS.csv"  # Your dataset containing hate and non-hate speech
    X, y = load_and_preprocess_data(csv_file)

    if X is not None and y is not None:
        # Split the dataset into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train the model
        model, vectorizer = train_decision_tree(X_train, y_train)

        # Evaluate the model
        X_test_tfidf = vectorizer.transform(X_test)
        y_pred = model.predict(X_test_tfidf)
        print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
        print(f"Classification Report:\n{classification_report(y_test, y_pred)}")

        # Output CSV file to store flagged comments
        output_file = "hateout.csv"

        # Loop to allow the user to input multiple comments
        while True:
            user_comment = input("Enter a comment to validate (or type 'exit' to quit): ")

            if user_comment.lower() == 'exit':
                print("Exiting the program.")
                break

            # Validate the comment using the trained model
            validate_user_input(model, vectorizer, user_comment, output_file)
