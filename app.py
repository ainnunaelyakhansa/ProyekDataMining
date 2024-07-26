from flask import Flask, request, render_template, jsonify
import pickle
import pandas as pd

app = Flask(__name__)

# Load model
model = pickle.load(open('models/model.pkl', 'rb'))

# Mapping dari indeks ke label
index_to_label = {
    0: 'Benign',
    1: 'Malignant'
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    df = pd.DataFrame([data])
    prediction = model.predict(df)
    # Ubah hasil prediksi dari angka ke label
    labels = [index_to_label[idx] for idx in prediction]
    return jsonify(labels)

if __name__ == "__main__":
    app.run(debug=True)