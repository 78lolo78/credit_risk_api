# Load librairies
import joblib
import pandas as pd
from flask import Flask, request, jsonify
import json
# -------------------------------------------------------------------------------------------
#                                           loadings
# -------------------------------------------------------------------------------------------
# Model loading
# -------------

bestmodel = joblib.load('scoring_credit_model.pkl')



# data loading
# -------------
X_test = joblib.load('X_test_sample_v2.csv')
X_train = joblib.load('X_train_sample_v2.csv')
y_test = joblib.load('y_test_sample_v2.csv')
y_train = joblib.load('y_train_sample_v2.csv')


x_train_sample = X_train.iloc[0: 100]
y_train_sample = y_train.iloc[0: 100]



X = pd.concat([X_test, X_train], axis=0)
y = pd.concat([y_test, y_train], axis=0)

# get the name of the columns
preproc_cols = X_train.columns

###############################################################
# instantiate Flask object
app = Flask(__name__)

@app.route("/")
def index():
    return "APP loaded, model and data loaded............"


# Customers id list  (
@app.route('/app/id/')
def ids_list():
    # Extract list of all the 'SK_ID_CURR' ids in the X_test dataframe
    customers_id_list = pd.Series(list(X.index.sort_values()))
    # Convert pd.Series to JSON
    customers_id_list_json = json.loads(customers_id_list.to_json())
    # Returning the processed data
    return jsonify({'status': 'ok',
                    'data': customers_id_list_json})


# test local : http://127.0.0.1:5000/app/data_cust/?SK_ID_CURR=165690
@app.route('/app/data_cust/')  # ==> OK
def selected_cust_data():  # selected_id
    selected_id_customer = int(request.args.get('SK_ID_CURR'))
    x_cust = X.loc[selected_id_customer: selected_id_customer]
    y_cust = y.loc[selected_id_customer: selected_id_customer]
    # Convert pd.Series to JSON
    data_x_json = json.loads(x_cust.to_json())
    y_cust_json = json.loads(y_cust.to_json())
    # Returning the processed data
    return jsonify({'status': 'ok',
                    'y_cust': y_cust_json,
                    'data': data_x_json})


# answer when asking for score and decision about one customer
# Test local : http://127.0.0.1:5000/app/scoring_cust/?SK_ID_CURR=165690

@app.route('/app/scoring_cust/')  # == > OK
def scoring_cust():
    # Parse http request to get arguments (sk_id_cust)
    customer_id = int(request.args.get('SK_ID_CURR'))
    # Get the data for the customer (pd.DataFrame)
    X_cust = X.loc[customer_id:customer_id]
    # X_cust = X_cust.drop(["SK_ID_CURR"], axis=1)
    # Compute the score of the customer (using the whole pipeline)
    score_cust = bestmodel.predict_proba(X_cust)[:, 1][0]
    # Return score
    return jsonify({'status': 'ok',
                    'SK_ID_CURR': customer_id,
                    'score': score_cust})

# answer when asking for decision about one customer
# Test local : http://127.0.0.1:5000/app/predict/?SK_ID_CURR=165690

@app.route('/app/predict/')  # == > OK
def predict():
    # Parse http request to get arguments (sk_id_cust)
    customer_id = int(request.args.get('SK_ID_CURR'))
    # Get the data for the customer (pd.DataFrame)
    X_cust = X.loc[customer_id:customer_id]
    # X_cust = X_cust.drop(["SK_ID_CURR"], axis=1)
    # Compute the score of the customer (using the whole pipeline)
    prediction = bestmodel.predict(X_cust)
    # Return answer
    if(prediction[0]>0.5):
        prediction="Refus"
    else:
        prediction="Accord"
    return jsonify({ 'SK_ID_CURR': customer_id,
                     'prediction': prediction})
    



# test
if __name__ == "__main__":
    app.run()
