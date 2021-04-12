from flask import Flask, redirect, url_for, render_template, request, session
from datetime import timedelta

app = Flask(__name__)
app.secret_key = "ThisIsNotAlabama"
app.permanent_session_lifetime = timedelta(days=5)

@app.route("/")
def index():
    if "user" in session:
        return render_template("index.html")
    else:
        return redirect(url_for("login"))

@app.route("/login", methods=["POST", "GET"])
def login():
    if "user" in session:
        return redirect(url_for("index"))
    else:
        if request.method == "POST":
            username = request.form["username"]
            password = request.form["password"]
            session.permanent = True
            session["user"] = username
            return redirect(url_for("index"))
        else:
            return render_template("login.html")

@app.route("/logout")
def logout():
    session.pop("user", None)
    return redirect(url_for("login"))

@app.route("/register", methods=["POST", "GET"])
def register():
    return render_template("register.html")


if __name__ == "__main__":
    app.run(debug=True)
