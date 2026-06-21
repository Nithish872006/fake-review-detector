from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

app = Flask(__name__)

# Load dataset
data = pd.read_csv("reviews.csv")

# Convert text into numbers
cv = CountVectorizer()
X = cv.fit_transform(data["review"])

# Labels
y = data["label"]

# Train model
model = MultinomialNB()
model.fit(X, y)

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = ""

    if request.method == "POST":
        review = request.form["review"]

        review_vector = cv.transform([review])

        result = model.predict(review_vector)

        prediction = result[0]

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)