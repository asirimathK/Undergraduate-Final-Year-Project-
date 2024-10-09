#phase 1 - manual labling testing
import pandas as pd
import os


# Function to check if any words in the user's comment exist in the CSV data
def validate_comment(csv_file, user_comment, output_file):
    try:
        # Read the CSV file into a DataFrame
        data = pd.read_csv(csv_file)

        # Assuming the CSV contains a column named 'text' with hate words
        if 'text' not in data.columns:
            print("Error: The CSV file must contain a 'text' column.")
            return

        # Convert all the words in the 'text' column to a set for faster lookup
        hate_words_set = set(data['text'].str.lower().values)

        # Split the user's comment into words and convert to lowercase for case-insensitive comparison
        comment_words = user_comment.lower().split()

        # Check if any word from the user's comment is in the hate words set
        found_words = [word for word in comment_words if word in hate_words_set]

        # Provide feedback based on the results
        if found_words:
            print(f"Found: Hateful or Offensive content: {', '.join(found_words)}")

            # Save the flagged comment to the output CSV file
            save_flagged_comment(output_file, user_comment)
        else:
            print("Success: No Hateful or Offensive content.")

    except FileNotFoundError:
        print("Error: CSV file not found. Please check the file path.")
    except Exception as e:
        print(f"An error occurred: {e}")


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
    # Example CSV file path containing hate words
    csv_file = "testD1.csv"

    # Output CSV file to store flagged comments
    output_file = "hateout.csv"

    while True:
        # Get user comment
        user_comment = input("Enter a comment to validate (or type 'exit' to quit): ")

        # Exit the loop if the user types 'exit'
        if user_comment.lower() == 'exit':
            print("Exiting the program.")
            break

        # Validate the comment against CSV data
        validate_comment(csv_file, user_comment, output_file)

#---------------------------------------------------------------------------------
#preprocessed inputs + flagged statements
import pandas as pd
import os
import re

# Function to clean and preprocess the user's comment
def preprocess_comment(comment):
    # Convert to lowercase
    comment = comment.lower()

    # Remove punctuation and special characters
    comment = re.sub(r'[^\w\s]', '', comment)

    # Split into words
    return comment.split()

# Function to check if any words in the user's comment exist in the CSV data
def validate_comment(csv_file, user_comment, output_file):
    try:
        # Read the CSV file into a DataFrame
        data = pd.read_csv(csv_file)

        # Check if required columns are present in the dataset
        if not {'text', 'category'}.issubset(data.columns):
            print("Error: The CSV file must contain 'text' and 'category' columns.")
            return

        # Convert all words in 'text' column to lowercase and create a set for faster lookup
        hate_words_set = set(data[data['category'] == 'hate']['text'].str.lower().values)
        offensive_words_set = set(data[data['category'] == 'offensive']['text'].str.lower().values)

        # Preprocess and tokenize the user's comment
        comment_words = preprocess_comment(user_comment)

        # Initialize categories
        found_hate_words = [word for word in comment_words if word in hate_words_set]
        found_offensive_words = [word for word in comment_words if word in offensive_words_set]

        # Categorize the statement
        if found_hate_words:
            print(f"Found: Hateful content: {', '.join(found_hate_words)}")
            category = "Hate Speech"
        elif found_offensive_words:
            print(f"Found: Offensive content: {', '.join(found_offensive_words)}")
            category = "Offensive"
        else:
            print("Success: No Hateful or Offensive content.")
            category = "Neither"

        # Save the flagged comment to the output CSV file
        save_flagged_comment(output_file, user_comment, category)

    except FileNotFoundError:
        print("Error: CSV file not found. Please check the file path.")
    except Exception as e:
        print(f"An error occurred: {e}")

# Function to save the flagged comment in the output CSV file
def save_flagged_comment(output_file, user_comment, category):
    # Create a DataFrame to store the flagged comment and its category
    df = pd.DataFrame({'statement': [user_comment], 'category': [category]})

    # Check if the output CSV file exists
    if os.path.isfile(output_file):
        # Append to the existing file
        df.to_csv(output_file, mode='a', header=False, index=False)
    else:
        # Create a new file with the header
        df.to_csv(output_file, mode='w', header=True, index=False)

    print(f"Flagged comment saved to {output_file} under the category: {category}")

# Main logic
if __name__ == "__main__":

    # Example CSV file path containing hate and offensive words with their categories
    csv_file = "data.csv"

    # Output CSV file to store flagged comments
    output_file = "hateout.csv"

    while True:
        # Get user comment
        user_comment = input("Enter a comment to validate (or type 'exit' to quit): ")

        # Exit the loop if the user types 'exit'
        if user_comment.lower() == 'exit':
            print("Exiting the program.")
            break

        # Validate the comment against CSV data
        validate_comment(csv_file, user_comment, output_file)
