import os
import eventlet
import flask
from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit
from langchain_huggingface import HuggingFaceEndpoint
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

huggingfacehub_api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

if not huggingfacehub_api_token:
    raise ValueError("HUGGINGFACEHUB_API_TOKEN not found in .env file")

llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    task="text-generation",
    temperature=0.5,
    max_new_tokens=600
)

# Flask App
app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# Dictionary to store chat history for each session
chat_histories = {}

@app.route("/")
def index():
    return render_template("index.html")

@socketio.on("send_message")
def handle_message(data):
    user_message = data["message"]
    session_id = request.sid  # Get the unique session ID

    # Initialize chat history for this session if it doesn't exist
    if session_id not in chat_histories:
        chat_histories[session_id] = [
            "You are Garry, an AI assistant specializing in code generation and technical help. "
            "Keep explanations clear and well-formatted. Use proper headings, spacing, and indentation. "
        ]

    chat_history = chat_histories[session_id]  # Get this user's chat history
    chat_history.append(f"User: {user_message}")

    try:
        context = "\n".join(chat_history[-5:])  # Use the last 5 messages for context
        ai_response = llm.invoke(f"{context}\nUser: {user_message}\nGarry:").strip()
        chat_history.append(f"Garry: {ai_response}")

    except Exception as e:
        error_message = str(e)
        ai_response = '''Our servers are facing a midlife crisis. Hang tight or maybe, 🔄 Refresh to help it find its way...
                                   
                                     ! Thank You ♥ For Your Patience !
        '''
        emit("log_error", {"error": error_message})

    # Send response back to the same user only
    emit("receive_message", {"user": user_message, "bot": ai_response})

@socketio.on("disconnect")
def handle_disconnect():
    """ Remove chat history when the user disconnects to free memory. """
    session_id = request.sid
    if session_id in chat_histories:
        del chat_histories[session_id]

if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=5000, debug=True)
