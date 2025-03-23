import numpy as np
from flask import Flask, request, render_template,jsonify
import pickle
from flask import redirect, url_for


app = Flask(__name__)

# Load the trained model dictionary
classifiers = pickle.load(open("model.pkl", "rb"))

# Extract the specific classifier you want to use for prediction
model = classifiers["XGBClassifier"]

# Define label mapping
label_mapping = {0: "NEGATIVE", 1: "HYPOTHYROID", 2: "HYPERTHYROID"}

# Define mapping for gender and Tsh measured
gender_mapping = {"F": 1, "M": 0}
tsh_measured_mapping = {"y": 1, "n": 0}
pregnant_mapping = {"y": 1, "n": 0}
t3_measured_mapping = {"y": 1, "n": 0}   
tt4_measured_mapping = {"y": 1, "n": 0}
t4u_measured_mapping = {"y": 1, "n": 0}

# @app.route('/result_page')
# def result_page():
#     prediction_text = request.args.get('prediction_text', '')
#     return render_template('result_page.html', prediction_text=prediction_text)
@app.route("/")
def home():
    return render_template("index.html")

app.config['static_FOLDER'] = 'static'
# app.config['UPLOAD_FOLDER'] = 'static'

@app.route("/predict", methods=["POST"])
def predict():
    # Get form data
    age = float(request.form.get("age"))
    gender = request.form.get("gender")
    pregnant = request.form.get("Pregnant","n")
    tsh_measured = request.form.get("Tsh measured")
    level = float(request.form.get("Level"))
    t3_measured = request.form.get("T3 measured")
    t3_level = float(request.form.get("T3 Level"))
    tt4 = request.form.get("Tt4")
    tt4_level = float(request.form.get("Tt4 level"))
    t4u = request.form.get("T4u")
    t4u_level = float(request.form.get("T4u level"))
    # fti = float(request.form.get("fti"))

    # Convert string values to numerical values
    gender_numeric = gender_mapping.get(gender, 0)
    pregnant_numeric = pregnant_mapping.get(pregnant, 0)
    tsh_measured_numeric = tsh_measured_mapping.get(tsh_measured, 0)
    # tsh_measured_numeric = tsh_measured_mapping.get(tsh_measured, 0)
    t3_measured_numeric = t3_measured_mapping.get(t3_measured, 0)
    tt4_numeric = tt4_measured_mapping.get(tt4, 0)
    t4u_numeric = t4u_measured_mapping.get(t4u, 0)

    if t3_level!=0:
        fti=(tt4_level/t3_level)*100
    else:
        fti=0

    # Create feature array
    features = np.array([age, gender_numeric, pregnant_numeric, tsh_measured_numeric, level, 
                         t3_measured_numeric, t3_level, tt4_numeric, tt4_level, t4u_numeric, t4u_level, fti]).reshape(1, -1)

    # Make prediction using the loaded model
    # prediction_label = model.predict(features)[0]

     # Make prediction using the loaded model
    prediction_label = model.predict(features)[0]

    # Map numerical label to string label
    # prediction_text = label_mapping[prediction_label]
    predicted_thyroid_type = label_mapping[prediction_label]

    # Redirect to the result page and pass the prediction_text as a query parameter
    return redirect(url_for('result_page', predicted_thyroid_type=predicted_thyroid_type))

@app.route('/result_page')
def result_page():
    # Get the prediction_text from the query parameters
    # prediction_text = request.args.get('prediction_text', '')
    predicted_thyroid_type = request.args.get('predicted_thyroid_type', '')
    prediction_text = f"Thyroid type is {predicted_thyroid_type}"
    return render_template('result_page.html', prediction_text=prediction_text)

    # return render_template('result_page.html', prediction_text=prediction_text)

    
    # return jsonify({"prediction_text": prediction_result})

    

if __name__ == "main":
    app.run(debug=True)