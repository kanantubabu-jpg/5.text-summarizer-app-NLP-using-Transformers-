from flask import Flask, render_template, request
from summarizer import abstractive_summarize, extractive_summarize

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def home():
    summary = ""
    original_text = ""
    selected_method = "transformers"
    fallback_message = ""

    if request.method == "POST":
        original_text = request.form.get("user_text", "")
        selected_method = request.form.get("method", "transformers")
        
        if original_text.strip():
            if selected_method == "transformers":
                summary = abstractive_summarize(original_text)
                if summary is None:
                    summary = extractive_summarize(original_text)
                    fallback_message = "Transformers is not installed, so an extractive summary was generated instead."
            else:
                summary = extractive_summarize(original_text)
        else:
            summary = "Please enter some text to summarize."

    return render_template(
        "index.html", 
        summary=summary, 
        original_text=original_text, 
        selected_method=selected_method,
        fallback_message=fallback_message
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)