from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

vectorizer = joblib.load("model/vectorizer.pkl")  
model_class = joblib.load("model/klasifikasi.pkl")
model_reg = joblib.load("model/regresi.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    description = data.get("description", "")

    if not description:
        return jsonify({"error": "Description is required"}), 400

    x = vectorizer.transform([description])
    y_class_pred = model_class.predict(x)
    y_reg_pred = model_reg.predict(x)

    response = {
        "objective": y_class_pred[0][0],
        "platform": y_class_pred[0][1],
        "industry": y_class_pred[0][2],
        "need_hosting": "Yes" if y_class_pred[0][3] == "1" else "No",
        "need_ui_ux": "Yes" if y_class_pred[0][4] == "1" else "No",
        "estimated_deadline": int(y_reg_pred[0][0]),
        "estimated_budget": int(y_reg_pred[0][1]),
    }

    return jsonify(response)

if __name__ == "__main__":
    app.run(debug=True)