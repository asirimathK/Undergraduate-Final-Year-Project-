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
            print(f"Your comment contains the following flagged word(s): {', '.join(found_words)}")

            # Save the flagged comment to the output CSV file
            save_flagged_comment(output_file, user_comment)
        else:
            print("Your comment is clean and contains no flagged words.")

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
