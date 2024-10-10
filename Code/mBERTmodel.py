import pandas as pd
import os
import torch
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tqdm import tqdm


# 1. Load Dataset
def load_data(file_path):
    # Load the dataset from CSV
    data = pd.read_csv(file_path)

    # Ensure there are no missing values
    data = data.dropna(subset=['text', 'label'])

    return data


# 2. Tokenization Function
def tokenize_texts(texts, tokenizer):
    return tokenizer(texts.tolist(), padding=True, truncation=True, return_tensors="pt", max_length=128)


# 3. Training Loop
def train(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    for batch in tqdm(dataloader):
        # Unpack the batch
        input_ids, attention_mask, labels = batch

        # Move inputs to the GPU if available
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)

        # Zero out any previously calculated gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()

        # Backward pass
        loss.backward()

        # Update weights
        optimizer.step()

    avg_loss = total_loss / len(dataloader)
    print(f"Average training loss: {avg_loss}")


# 4. Evaluation Function
def evaluate(model, X_test_tokens, y_test_tensor, device):
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids=X_test_tokens['input_ids'].to(device),
                        attention_mask=X_test_tokens['attention_mask'].to(device))

        # Get the predicted labels
        predictions = torch.argmax(outputs.logits, dim=1)

    predictions = predictions.cpu().numpy()
    y_test = y_test_tensor.cpu().numpy()

    print(classification_report(y_test, predictions, target_names=['Non-Hate', 'Hate']))


# 5. Save flagged hate speech comments to CSV
def save_hate_comment(user_comment, prediction, output_file):
    if prediction == 1:  # 1 corresponds to hate speech
        df = pd.DataFrame({'statement': [user_comment]})

        if os.path.isfile(output_file):
            df.to_csv(output_file, mode='a', header=False, index=False)
        else:
            df.to_csv(output_file, mode='w', header=True, index=False)


# 6. Predict and save function
def predict_and_save(user_comment, model, tokenizer, output_file, device):
    inputs = tokenizer(user_comment, return_tensors="pt", truncation=True, padding=True, max_length=128).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        prediction = torch.argmax(outputs.logits, dim=1).item()

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

    # Split into train/test sets
    X_train, X_test, y_train, y_test = train_test_split(data['text'], data['label'], test_size=0.2, random_state=42)

    # Load mBERT tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

    # Tokenize train and test sets
    X_train_tokens = tokenize_texts(X_train, tokenizer)
    X_test_tokens = tokenize_texts(X_test, tokenizer)

    # Load mBERT model
    model = BertForSequenceClassification.from_pretrained('bert-base-multilingual-cased', num_labels=2)

    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Prepare labels
    y_train_tensor = torch.tensor(y_train.values).to(device)
    y_test_tensor = torch.tensor(y_test.values).to(device)

    # Create DataLoader for training
    train_dataset = TensorDataset(X_train_tokens['input_ids'], X_train_tokens['attention_mask'], y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    # Optimizer
    optimizer = AdamW(model.parameters(), lr=2e-5)

    # Train the model for 3 epochs
    for epoch in range(3):
        print(f"Epoch {epoch + 1}")
        train(model, train_loader, optimizer, device)

    # Evaluate the model on the test set
    evaluate(model, X_test_tokens, y_test_tensor, device)

    # Predict and save flagged comments
    output_file = "hateout.csv"
    while True:
        user_comment = input("Enter a comment (or type 'exit' to quit): ")
        if user_comment.lower() == 'exit':
            break
        predict_and_save(user_comment, model, tokenizer, output_file, device)
