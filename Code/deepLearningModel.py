import pandas as pd
import os
import numpy as np
import re
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping


# 1. Load Dataset
def load_data(file_path):
    # Load the dataset from CSV
    data = pd.read_csv(file_path)

    # Drop rows with missing values
    data = data.dropna(subset=['text', 'label'])

    return data


# 2. Text Preprocessing
def preprocess_text(text):
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)  # Remove URLs
    text = re.sub(r'\@w+|\#', '', text)  # Remove mentions and hashtags
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = text.lower()  # Convert text to lowercase
    return text


# 3. Prepare Data for Deep Learning
def prepare_data(data, tokenizer, max_len):
    # Tokenize the input text
    sequences = tokenizer.texts_to_sequences(data)

    # Pad sequences to ensure equal length
    padded_sequences = pad_sequences(sequences, maxlen=max_len, padding='post')

    return padded_sequences


# 4. Build LSTM Model
def build_lstm_model(vocab_size, embedding_dim, max_len):
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_len),
        LSTM(units=128, return_sequences=False),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='sigmoid')  # Binary classification: hate or not hate
    ])

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# 5. Train the Model
def train_model(model, X_train, y_train, X_val, y_val, epochs, batch_size):
    # Early stopping callback
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping]
    )
    return model


# 6. Evaluation Function
def evaluate_model(model, X_test, y_test):
    # Evaluate the model on the test set
    predictions = (model.predict(X_test) > 0.5).astype(int)
    print(classification_report(y_test, predictions, target_names=['Non-Hate', 'Hate']))


# 7. Save flagged hate speech comments to CSV
def save_hate_comment(user_comment, prediction, output_file):
    if prediction == 1:  # 1 corresponds to hate speech
        df = pd.DataFrame({'statement': [user_comment]})

        if os.path.isfile(output_file):
            df.to_csv(output_file, mode='a', header=False, index=False)
        else:
            df.to_csv(output_file, mode='w', header=True, index=False)


# 8. Predict and save function
def predict_and_save(user_comment, model, tokenizer, max_len, output_file):
    user_comment_processed = preprocess_text(user_comment)
    sequence = tokenizer.texts_to_sequences([user_comment_processed])
    padded_sequence = pad_sequences(sequence, maxlen=max_len, padding='post')

    # Predict the class
    prediction = (model.predict(padded_sequence) > 0.5).astype(int)[0][0]

    if prediction == 1:
        print("Hate Speech detected!")
        save_hate_comment(user_comment, prediction, output_file)
    else:
        print("No Hate Speech detected.")


# Main Function
if __name__ == "__main__":
    # Load and preprocess dataset
    file_path = "fypDS.csv"  # Path to your dataset
    data = load_data(file_path)

    # Preprocess the text data
    data['text'] = data['text'].apply(preprocess_text)

    # Split data into train, validation, and test sets
    X_train, X_test, y_train, y_test = train_test_split(data['text'], data['label'], test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    # Initialize tokenizer and fit on training data
    vocab_size = 10000
    tokenizer = Tokenizer(num_words=vocab_size, oov_token='<OOV>')
    tokenizer.fit_on_texts(X_train)

    # Max length of sequences for padding
    max_len = 100

    # Prepare training, validation, and test data
    X_train_seq = prepare_data(X_train, tokenizer, max_len)
    X_val_seq = prepare_data(X_val, tokenizer, max_len)
    X_test_seq = prepare_data(X_test, tokenizer, max_len)

    # Build LSTM model
    embedding_dim = 128
    model = build_lstm_model(vocab_size, embedding_dim, max_len)

    # Train the model
    epochs = 10
    batch_size = 32
    model = train_model(model, X_train_seq, y_train, X_val_seq, y_val, epochs, batch_size)

    # Evaluate the model on the test set
    evaluate_model(model, X_test_seq, y_test)

    # Predict and save flagged comments
    output_file = "hateout.csv"
    while True:
        user_comment = input("Enter a comment (or type 'exit' to quit): ")
        if user_comment.lower() == 'exit':
            break
        predict_and_save(user_comment, model, tokenizer, max_len, output_file)
