from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
from agents import analyze_dream

app = Flask(__name__, static_folder=".")
CORS(app)

@app.route("/")
def index():
    return send_from_directory(".", "index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json()
    dream_text = (data or {}).get("dream", "").strip()
    api_key    = (data or {}).get("api_key", "").strip()

    if not dream_text:
        return jsonify({"error": "Please enter your dream."}), 400
    if not api_key:
        return jsonify({"error": "Please enter your Gemini API key."}), 400

    try:
        result = analyze_dream(dream_text, api_key)
        return jsonify(result)
    except Exception as e:
        msg = str(e)
        if "API_KEY" in msg.upper() or "credentials" in msg.lower():
            return jsonify({"error": "Invalid API key. Check your Gemini API key."}), 401
        return jsonify({"error": f"Agent error: {msg}"}), 500

if __name__ == "__main__":
    print("🌙 Dream Analyzer running at http://0.0.0.0:5000")
    app.run(host="0.0.0.0", port=5000, debug=False)