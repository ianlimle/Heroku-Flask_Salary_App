from flask import Flask, request, jsonify, render_template
import numpy as np
import pickle

# assigning Flask constructor to an variable app
app = Flask(__name__)
# load the model
model = pickle.load(open('model.pkl', 'rb'))

# route to the home function as defined below which will return the index.html page that is the home page
@app.route('/')
def home():
    # render the home page according to the index.html file
    return render_template('index.html')

# route to the predict function below
@app.route('/predict', methods=['POST'])
def predict():
    #retrieve the values in respective text fields
    features = [int(x) for x in request.form.values()]
    feat_arr = [np.array(features)]
    pred = model.predict(feat_arr)
    output = round(pred[0], 2)
    
    return render_template('index.html', pred_text='The employee salary prediction is ${}'.format(output))

if __name__=="__main__":
    app.run(debug=True)
