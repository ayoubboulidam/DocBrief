import os
from dotenv import load_dotenv
from flask import Flask, request, render_template, jsonify
from services.document_processing import process_pdf
from services.summary_generation import get_summary

# Load environment variables from .env file
load_dotenv()

# Initialize the Flask app
app = Flask(__name__)

@app.route('/')
def index():
    # Render the main page (index.html)
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    # Check if a file is included in the request
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    # Get the file from the request
    file = request.files['file']

    # Ensure the file has a name
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    try:
        # Process the uploaded PDF and create the vector store
        vectorstore = process_pdf(file)

        # Query the vector store for a document summary
        query = "Give me the gist of the document in 3 sentences."
        summary = get_summary(query, vectorstore)

        # Return the summary as a JSON response
        return jsonify({"summary": summary})

    except Exception as e:
        # Handle any errors and return them in the response
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    # Run the Flask app in debug mode
    app.run(debug=True)