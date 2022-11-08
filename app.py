import json
from flask import Flask, render_template ,jsonify
import os
from flask import request

import numpy as np
import pickle


project_dir = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__ , template_folder="templates")
model = pickle.load(open("model.pkl", "rb"))

@app.route('/', methods=['POST','GET'])
def home():
    return render_template("index.html")



def isfloat(num):
    try:
        float(num)
        return True
    except ValueError:
        return False
@app.route("/predict", methods = ["POST"])
def predict():
    float_features = []
    for x in request.form.values():
        if not isfloat(x):
            response = {"status" : 500,"status_msg": "Some fields are empty !"}
            return jsonify(response)

        float_features.append(float(x))


    final = [np.array(float_features)]
    prediction = model.predict(final)

    response = {"status" : 200,"status_msg": "Student may Fail !!!"}


    if prediction == "Pass":
        response = {"status" : 200,"status_msg": "Student will Pass !!!"}
      
    return jsonify(response)
if __name__ == '__main__':
    app.run(debug=True)