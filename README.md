# ML_as_service
Expose a Machine Learning Application(Classification) as a REST API service using python flask

* The file app.py has two main functions
** get_prediction**
*- receives the request as a json object
** model()**
*- Performs all the pre-processing and feature engineering 
*- Inputs all required variables to the Logistic regression equation.
*- Model threshold is fixed at 0.7

# Execution Steps:
* Navigate to the directory of source code and execute the app.py. By default, the application starts listening http://127.0.0.1:5000/predict.
* Open another terminal to post  request to the API
** requests.post('http://127.0.0.1:5000/predict',json = {})**
* on successfull invocation of the service, you get the prediction as either 0 or 1.

# Sample JSON Request:
{"v1":"a", "v2":"17,92", "v3":"5,4e-05","v4":"u","v5":"g","v6":"c","v7":"v","v8":"1,75", "v9":"f"
, "v10":"t","v11":1,"v12":"t", "v13":"g", "v14":"80.0","v15":5,"v17":"800000.0"	,"v18":"t","v19":0}


