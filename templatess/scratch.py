from flask import Flask, render_template, request, redirect, url_for
import os

app = Flask(__name__)

# File to save comments
comments_file = "comments.txt"

# Ensure the file exists
if not os.path.exists(comments_file):
    with open(comments_file, "w") as file:
        pass

@app.route('/')
def index():
    # Read all the comments from the file
    with open(comments_file, "r") as file:
        comments = file.readlines()
    return render_template('index.html', comments=comments)

@app.route('/submit', methods=['POST'])
def submit_comment():
    # Get the user's comment from the form
    comment = request.form.get('comment')

    # If the cancel button was clicked, do nothing and redirect back to the main page
    if request.form['action'] == 'Cancel':
        return redirect(url_for('index'))

    # Save the comment to the file if the submit button was clicked
    if comment.strip():  # Only save if the comment is not empty
        with open(comments_file, "a") as file:
            file.write(comment + "\n")

    # Redirect back to the main page
    return redirect(url_for('index'))

if __name__ == "__main__":
    app.run(debug=True)
