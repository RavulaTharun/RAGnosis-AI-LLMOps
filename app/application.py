from flask import Flask, render_template, request, session, redirect, url_for
from app.components.retriever import create_qa_chain
from dotenv import load_dotenv
from markupsafe import Markup
import os, traceback

# Load environment variables
load_dotenv()
app = Flask(__name__)
app.secret_key = os.urandom(24)


# Custom Jinja filter for displaying line breaks
def nl2br(value):
    if not value:
        return ""
    return Markup(value.replace("\n", "<br>\n"))

app.jinja_env.filters["nl2br"] = nl2br


@app.route("/", methods=["GET", "POST"])
def index():
    if "messages" not in session:
        session["messages"] = []

    if request.method == "POST":
        user_input = request.form.get("prompt")

        if user_input:
            messages = session["messages"]
            messages.append({"role": "user", "content": user_input})
            session["messages"] = messages

            try:
                qa_chain = create_qa_chain()
                if qa_chain is None:
                    raise Exception("QA chain could not be created (LLM or VectorStore issue)")

                # ✅ FIXED KEY — use 'question' instead of 'input'
                # ✅ FINAL FIX — provide both keys to avoid KeyError
                response = qa_chain.invoke({
                    "input": user_input,      # required by LangChain retrieval chain
                    "question": user_input    # used by your custom prompt
                })


                # ✅ Extract the answer safely
                if isinstance(response, dict):
                    result = (
                        response.get("answer")
                        or response.get("result")
                        or response.get("output_text")
                        or str(response)
                    )
                else:
                    result = str(response)

                # Add assistant response
                messages.append({"role": "assistant", "content": result})
                session["messages"] = messages

            except Exception as e:
                print("\n🔥 Exception while running QA chain:")
                traceback.print_exc()
                error_msg = f"Error : {str(e)}"
                return render_template("index.html", messages=session["messages"], error=error_msg)

        return redirect(url_for("index"))

    return render_template("index.html", messages=session.get("messages", []))


@app.route("/clear")
def clear():
    session.pop("messages", None)
    return redirect(url_for("index"))


if __name__ == "__main__":
    print("🧭 Running from file:", __file__)
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)
