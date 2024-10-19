import pandas as pd
from transformers import BertTokenizer, BertModel
import torch
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import os

# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Function to get BERT embeddings
def get_bert_embeddings(text):
    # Tokenize the input text and get BERT embeddings
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    # Get the embeddings of the [CLS] token (for classification tasks)
    cls_embedding = outputs.last_hidden_state[:, 0, :].numpy()
    return cls_embedding

# Function to train SVM model using BERT embeddings
def train_svm_with_bert(X_train, y_train):
    # Fill missing values in the training data with empty strings
    X_train = X_train.fillna('')

    # Convert all text data into BERT embeddings
    X_train_bert = [get_bert_embeddings(text) for text in X_train]

    # Reshape the list of arrays into a matrix
    X_train_bert = torch.cat([torch.tensor(emb) for emb in X_train_bert], dim=0).numpy()

    # Initialize Support Vector Classifier
    svm_model = SVC(kernel='linear', random_state=42)

    # Train the SVM model
    svm_model.fit(X_train_bert, y_train)

    return svm_model

# Function to predict and evaluate the model
def evaluate_model_with_bert(svm_model, X_test, y_test):
    # Fill missing values in the test data with empty strings
    X_test = X_test.fillna('')

    # Convert test data into BERT embeddings
    X_test_bert = [get_bert_embeddings(text) for text in X_test]
    X_test_bert = torch.cat([torch.tensor(emb) for emb in X_test_bert], dim=0).numpy()

    # Predict the labels for the test data
    y_pred = svm_model.predict(X_test_bert)

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
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the SVM model with BERT embeddings
    svm_model = train_svm_with_bert(X_train, y_train)

    # Evaluate the model on the test data
    evaluate_model_with_bert(svm_model, X_test, y_test)

    # Allow user to input a comment and validate it
    while True:
        user_comment = input("Enter a comment to validate (or type 'exit' to quit): ")

        if user_comment.lower() == 'exit':
            print("Exiting the program.")
            break

        # Preprocess and validate the user input using BERT embeddings
        user_comment_bert = get_bert_embeddings([user_comment])
        prediction = svm_model.predict(user_comment_bert)

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
