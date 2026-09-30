from flask import Flask, request, jsonify
app = Flask(__name__)

@app.route('/')
def home():
    return "Credit Scoring API is Running"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    if data['Credit_Score'] > 650 and data['Income'] > 40000:
        return jsonify({"result": "Loan Approved", "risk": "Low"})
    else:
        return jsonify({"result": "Loan Rejected", "risk": "High"})

if __name__ == '__main__':
    app.run(debug=True)
