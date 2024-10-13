import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, MaxPooling1D, LSTM, Dense, Dropout, GlobalMaxPooling1D
from tensorflow.keras.callbacks import EarlyStopping


# Load your dataset
def load_data(csv_file):
    data = pd.read_csv(csv_file)
    # Assuming the dataset has 'text' and 'label' columns
    X = data['text']
    y = data['label']
    return X, y


# Preprocess the text (tokenization and padding)
def preprocess_data(X, max_len=100, max_words=10000):
    tokenizer = Tokenizer(num_words=max_words, lower=True)
    tokenizer.fit_on_texts(X)
    sequences = tokenizer.texts_to_sequences(X)
    X_pad = pad_sequences(sequences, maxlen=max_len)
    return X_pad, tokenizer


# Build CNN-LSTM model
def build_cnn_lstm_model(input_length, vocab_size, embedding_dim=128):
    model = Sequential()

    # Embedding Layer
    model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=input_length))

    # CNN Layer
    model.add(Conv1D(filters=128, kernel_size=5, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))

    # LSTM Layer
    model.add(LSTM(128, return_sequences=True))
    model.add(GlobalMaxPooling1D())

    # Dense Layers
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))

    # Output Layer (binary classification: hate or non-hate)
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


# Train the model
def train_model(model, X_train, y_train, X_val, y_val, epochs=10, batch_size=32):
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=batch_size,
                        callbacks=[early_stopping])
    return model, history


# Save flagged hate speech to output file
def save_flagged_comment(output_file, user_comment):
    df = pd.DataFrame({'statement': [user_comment]})
    if os.path.isfile(output_file):  # Check if the output file exists
        df.to_csv(output_file, mode='a', header=False, index=False)
    else:
        df.to_csv(output_file, mode='w', header=True, index=False)


# Evaluate the model
def evaluate_model(model, X_test, y_test):
    # Make predictions on the test set
    y_pred = (model.predict(X_test) > 0.5).astype("int32")

    # Accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")

    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Confusion matrix
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))


# Main logic for user input
def main():
    # Load and preprocess the data
    csv_file = "fypDS.csv"  # Your dataset with 'text' and 'label' columns
    output_file = "hateout.csv"  # Output file to store flagged comments

    X, y = load_data(csv_file)
    X_pad, tokenizer = preprocess_data(X)

    # Split the data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X_pad, y, test_size=0.2, random_state=42)

    # Build and train the CNN-LSTM model
    model = build_cnn_lstm_model(input_length=X_pad.shape[1], vocab_size=len(tokenizer.word_index) + 1)
    model, history = train_model(model, X_train, y_train, X_val, y_val, epochs=10)

    # Evaluate the model
    print("\nEvaluating the model on the validation set:")
    evaluate_model(model, X_val, y_val)

    # User input and hate speech detection
    while True:
        user_comment = input("Enter a comment to validate (or type 'exit' to quit): ")

        if user_comment.lower() == 'exit':
            print("Exiting the program.")
            break

        # Preprocess the user's comment
        user_comment_seq = tokenizer.texts_to_sequences([user_comment])
        user_comment_pad = pad_sequences(user_comment_seq, maxlen=X_pad.shape[1])

        # Predict whether the comment is hate speech or not
        prediction = model.predict(user_comment_pad)[0][0]

        if prediction > 0.5:
            print("Hate Speech Detected!")
            save_flagged_comment(output_file, user_comment)
        else:
            print("No Hate Speech Detected.")

    print("Model training and evaluation completed.")


if __name__ == "__main__":
    main()
