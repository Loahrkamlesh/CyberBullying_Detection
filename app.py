from flask import Flask, render_template, request
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)

# Load vocabulary dictionary
with open("tfidfvectoizer.pkl", "rb") as f:
    vocab = pickle.load(f)

# Create vectorizer using vocabulary
vectorizer = TfidfVectorizer(vocabulary=vocab)

# Load trained model
with open("LinearSVCTuned.pkl", "rb") as f:
    model = pickle.load(f)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None

    if request.method == 'POST':
        user_input = request.form['text']

        transformed_input = vectorizer.fit_transform([user_input])

        prediction = model.predict(transformed_input)[0]

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
    