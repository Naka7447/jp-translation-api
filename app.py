import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

app = Flask(__name__)
CORS(app)

print("Loading NLLB model... this happens once when the server starts.")
model_name_nllb = "facebook/nllb-200-distilled-600M"
tokenizer_nllb = AutoTokenizer.from_pretrained(model_name_nllb)
model_nllb = AutoModelForSeq2SeqLM.from_pretrained(model_name_nllb)
print("Model loaded. Server ready.")


def translate(text, src_lang, tgt_lang):
    tokenizer_nllb.src_lang = src_lang
    inputs = tokenizer_nllb(text, return_tensors="pt")
    tokens = model_nllb.generate(
        **inputs,
        forced_bos_token_id=tokenizer_nllb.convert_tokens_to_ids(tgt_lang)
    )
    return tokenizer_nllb.batch_decode(tokens, skip_special_tokens=True)[0]


@app.route("/translate", methods=["POST"])
def translate_endpoint():
    data = request.get_json()

    if not data or "text" not in data or "direction" not in data:
        return jsonify({"error": "Request must include 'text' and 'direction'"}), 400

    text = data["text"]
    direction = data["direction"]

    if direction == "ja-en":
        result = translate(text, "jpn_Jpan", "eng_Latn")
    elif direction == "en-ja":
        result = translate(text, "eng_Latn", "jpn_Jpan")
    else:
        return jsonify({"error": "direction must be 'ja-en' or 'en-ja'"}), 400

    return jsonify({"translation": result})


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port)
