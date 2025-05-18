from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load model and label encoder
model = pickle.load(open("model.pkl", "rb"))
label_encoder = pickle.load(open("label_encoder.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        math = float(request.form["math"])
        reading = float(request.form["reading"])
        writing = float(request.form["writing"])
        input_data = np.array([[math, reading, writing]])

        prediction = model.predict(input_data)[0]
        label = label_encoder.inverse_transform([prediction])[0]

        return render_template("index.html", prediction_text=f"Predicted Race/Ethnicity: {label}")
    except:
        return render_template("index.html", prediction_text="Error: please enter valid numeric scores.")

if __name__ == "__main__":
    app.run(debug=True)
