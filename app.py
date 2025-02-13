import os
import eventlet
import flask
from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit
from langchain_huggingface import HuggingFaceEndpoint
from dotenv import load_dotenv  # Add this import

# Load environment variables first
load_dotenv()  # Load from .env file

# Configure AI Model
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

chat_history = [
    "You are Garry, an AI assistant specializing in code generation and technical help. "
    "Keep explanations clear and well-formatted and most importantly complete. Use proper headings, spacing, and indentation in both text and code. "
    "Ensure line breaks between paragraphs and maintain readable formatting. "
    "Always use proper headings, bullet points, line breaks, and indentation for readability."
]

@app.route("/")
def index():
    return render_template("index.html")

@socketio.on("send_message")
def handle_message(data):
    user_message = data["message"]
    chat_history.append(f"User: {user_message}")

    try:
        context = "\n".join(chat_history[-5:])  # Last 5 messages for context
        ai_response = llm.invoke(f"{context}\nUser: {user_message}\nGarry:").strip()
        chat_history.append(f"Garry: {ai_response}")

    except Exception as e:
        error_message = str(e)
        ai_response = '''Our servers are facing a midlife crisis. Hang tight or maybe, 🔄 Refresh to help it find its way...
                                   
                                     ! Thank You ♥ For Your Patience !
        '''
        emit("log_error", {"error": error_message})

    emit("receive_message", {"user": user_message, "bot": ai_response}, broadcast=True)

if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=5000, debug=True)