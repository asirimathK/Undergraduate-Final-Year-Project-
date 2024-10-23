import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle

# Load your dataset
data = pd.read_csv("fypDS.csv")  # Ensure this dataset is available

# Assuming 'text' is the feature and 'label' is the target
X = data['text']
y = data['label']

# Fill missing values in the dataset
X = X.fillna('')

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Initialize TfidfVectorizer
vectorizer = TfidfVectorizer()

# Fit and transform the training data
X_train_tfidf = vectorizer.fit_transform(X_train)

# Initialize the Random Forest Classifier
model = RandomForestClassifier(random_state=42)

# Train the Random Forest model
model.fit(X_train_tfidf, y_train)

# Save the trained model and the vectorizer as .pkl files
with open("random_forest_model.pkl", "wb") as model_file:
    pickle.dump(model, model_file)

with open("tfidf_vectorizer.pkl", "wb") as vectorizer_file:
    pickle.dump(vectorizer, vectorizer_file)

print("Model and vectorizer have been saved successfully.")
