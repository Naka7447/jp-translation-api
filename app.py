"""
Kikoeru Translation API
A small Flask server that forwards translation requests to a community-hosted
NLLB API (winstxnhdw/nllb-api on Hugging Face Spaces), avoiding the need to
load the model locally (which exceeds free-tier hosting memory limits).
"""

import os
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

NLLB_API_BASE = "https://winstxnhdw-nllb-api.hf.space/api/v4/translator"

LANG_CODES = {
    "ja": "jpn_Jpan",
    "en": "eng_Latn",
}


@app.route("/translate", methods=["POST"])
def translate_endpoint():
    """
    Expects JSON body: { "text": "...", "direction": "ja-en" or "en-ja" }
    Returns JSON: { "translation": "..." }
    """
    data = request.get_json()

    if not data or "text" not in data or "direction" not in data:
        return jsonify({"error": "Request must include 'text' and 'direction'"}), 400

    text = data["text"]
    direction = data["direction"]

    if direction == "ja-en":
        source, target = LANG_CODES["ja"], LANG_CODES["en"]
    elif direction == "en-ja":
        source, target = LANG_CODES["en"], LANG_CODES["ja"]
    else:
        return jsonify({"error": "direction must be 'ja-en' or 'en-ja'"}), 400

    try:
        response = requests.get(
            NLLB_API_BASE,
            params={"text": text, "source": source, "target": target},
            timeout=15
        )
        response.raise_for_status()
        result = response.json()
        # The community API's exact response shape - adjust key if needed
        translation = result.get("result") or result.get("translation") or str(result)
        return jsonify({"translation": translation})
    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"Translation service unavailable: {str(e)}"}), 502


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port)
