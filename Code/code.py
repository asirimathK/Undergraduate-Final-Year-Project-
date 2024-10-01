import pandas as pd

# Function to check user input in CSV data
def validate_user_input(csv_file, user_input):
    try:
        # Read the CSV file into a DataFrame
        data = pd.read_csv(csv_file)

        # Assuming the CSV contains a column named 'text'
        if 'text' not in data.columns:
            print("Error: The CSV file must contain a 'text' column.")
            return

        # Check if user input exists in the 'text' column
        if user_input in data['text'].values:
            print(f"'{user_input}' is available in the CSV file.")
        else:
            print(f"'{user_input}' is NOT available in the CSV file.")
    except FileNotFoundError:
        print("Error: CSV file not found. Please check the file path.")
    except Exception as e:
        print(f"An error occurred: {e}")


# Main logic
if __name__ == "__main__":
    # Example CSV file path
    csv_file = "testD1.csv"

    while True:
        # Get user input
        user_input = input("Enter text to validate : ")

        # Exit the loop if the user types 'exit'
        if user_input.lower() == 'exit':
            print("Exiting the program.")
            break

        # Validate the input against CSV data
        validate_user_input(csv_file, user_input)

