from flask import Flask, render_template, request, jsonify, session
from flask_cors import CORS
import pandas as pd
import random
import re
import nltk
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer, util

# Download stopwords if not already downloaded
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Hardcoded responses for common phrases
hardcoded_responses = {
    "hi": "Hey! How's it going?",
    "hello": "Hey there! How can I help you?",
    "hey": "Hey! What’s on your mind?",
    "how are you": "I’m doing great! How about you?",
    "good morning": "Morning! Hope your day’s going well.",
    "good evening": "Good evening! How’s your day been?",
}

# Contraction mapping for more natural input processing
contraction_mapping = {
    "i'm": "i am",
    "you're": "you are",
    "he's": "he is",
    "she's": "she is",
    "it's": "it is",
    "they're": "they are",
    "we're": "we are",
    "isn't": "is not",
    "aren't": "are not",
    "wasn't": "was not",
    "weren't": "were not",
    "don't": "do not",
    "doesn't": "does not",
    "didn't": "did not",
    "can't": "cannot",
    "couldn't": "could not",
    "shouldn't": "should not",
    "won't": "will not",
    "wouldn't": "would not",
    "haven't": "have not",
    "hasn't": "has not",
    "hadn't": "had not"
}

def preprocess_text(text):
    """Cleans text, expands contractions, and removes stopwords for better matching."""
    text = text.lower().strip()

    # Expand contractions
    for contraction, full_form in contraction_mapping.items():
        text = text.replace(contraction, full_form)

    # Keep common greetings unchanged
    if text in hardcoded_responses:
        return text  

    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Remove special characters
    words = text.split()
    words = [word for word in words if word not in stop_words]  # Remove stopwords
    return ' '.join(words)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes
app.secret_key = "your_secret_key"  # Required for session tracking

# Load the dataset
file_path ="data.csv"
df = pd.read_csv(file_path)

# Ensure required columns exist
if 'User Query' not in df.columns or 'Bot Response' not in df.columns:
    raise ValueError("Dataset must contain 'User Query' and 'Bot Response' columns.")

df['Processed Query'] = df['User Query'].apply(preprocess_text)

# Load an improved semantic matching model (Paraphrase-aware)
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
query_embeddings = model.encode(df['Processed Query'].tolist(), convert_to_tensor=True)

def get_response(user_input):
    """Handles conversational language and provides relevant responses."""
    user_input = preprocess_text(user_input)

    # Check if user is responding to a follow-up question
    last_bot_question = session.get("last_bot_question", None)
    if last_bot_question:
        session["last_bot_question"] = None  # Clear after receiving a response
        return handle_follow_up(user_input, last_bot_question)

    # Check for direct responses
    if user_input in hardcoded_responses:
        return hardcoded_responses[user_input]

    # Handle empty or unclear inputs
    if not user_input.strip():
        return "Could you clarify that for me?"

    # Use semantic similarity for better matching
    user_embedding = model.encode(user_input, convert_to_tensor=True)
    similarities = util.pytorch_cos_sim(user_embedding, query_embeddings).flatten()

    top_index = similarities.argmax().item()  
    top_score = similarities[top_index].item()

    print(f"Top similarity score: {top_score}")  # Debugging log

    dynamic_threshold = 0.3  # Slightly relaxed threshold

    if top_score > dynamic_threshold:
        bot_response = df.iloc[top_index]['Bot Response']
    else:
        bot_response = "Hmm, I’m not sure I understand. Could you explain a little more?"

    # Store context for follow-ups if needed
    if bot_response.endswith("?"):  # If the bot response is a question, save it
        session["last_bot_question"] = bot_response

    return bot_response

def handle_follow_up(user_input, last_bot_question):
    """Provides meaningful follow-ups based on the user’s previous response."""
    if "stress" in last_bot_question.lower():
        return "I hear you. Have you tried taking a break or talking to someone about it?"

    if "feeling down" in last_bot_question.lower():
        return "I’m really sorry you’re feeling that way. Do you want to talk about what’s on your mind?"

    return "Thanks for sharing that. I'm here to listen."

@app.route("/")
def home():
    session.clear()  # Reset conversation on a new session
    return render_template("index.html")

@app.route("/get_response", methods=["POST"])
def chatbot_response():
    """Handles chatbot API requests."""
    data = request.get_json()
    user_message = data.get("message", "")

    # Get the bot's response
    bot_reply = get_response(user_message)

    return jsonify({"response": bot_reply})

if __name__ == "__main__":
    app.run(debug=True)



