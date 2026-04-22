from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

model = pickle.load(open("model.pkl","rb"))
cv = pickle.load(open("vectorizer.pkl","rb"))

@app.route("/", methods=["GET","POST"])
def home():
    prediction = ""

    if request.method == "POST":
        message = request.form["message"]
        data = cv.transform([message])
        result = model.predict(data)

        if result[0] == "spam":
            prediction = "⚠ This is SPAM Message"
        else:
            prediction = "✅ This is NOT Spam"

    return render_template("index.html", prediction=prediction)

if __name__=="__main__":
    app.run(debug=True)