from flask import Flask, render_template, request, jsonify
from wtforms import Form, TextAreaField, validators
import numpy as np
import pandas as pd
import json

app = Flask(__name__)


def model(obs):
    '''
    This function reads the json request as an argument
    All preprocessing and Feature Engineering for modelling are done.
    '''
    df = pd.DataFrame(obs, index = [0])
    df.drop('v18', axis  = 1, inplace = 1)
    df_1 = pd.concat([df
                  ,df['v2'].str.split(',', expand=True)
                  ,df['v3'].str.split(',', expand=True)
                  ,df['v8'].str.split(',', expand=True)],axis=1)
    print(df_1.head())
    print(df_1.columns)
    df_1.drop(['v2','v3','v8'], axis = 1, inplace = 1)
    df_1.columns = [ 'v1','v10','v11','v12','v13','v14'\
                    , 'v15', 'v17', 'v19',  'v2'\
                    ,'v5',  'v6',  'v7', 'v9',\
                    'v21','v22','v31','v32','v81','v82']
    df_1[['v21','v22','v31','v32','v81','v82']] = df_1[['v21','v22','v31','v32','v81','v82']].astype(float, errors = 'coerce')
    df_1[['v22','v31','v32','v81','v82']] = df_1[['v22','v31','v32','v81','v82']].apply(lambda x: x.fillna(0),axis = 1)
    
    # transformation of Numeric Variables
    # v21
    df_1['log_v21'] = np.log(df_1['v21'].astype(float)+1)
    df_1['v22_flag'] = df_1['v22'].astype(float).apply(lambda x: 1 if x < 50 else 0)
    df_1['log_v82'] = np.log(df_1['v82'].astype(float)+1)
    df_1['log_v14'] = np.log(df_1['v14'].astype(float)+1)
    df_1['log_v15'] = np.log(df_1['v15'].astype(float)+1)
    
    # Initialize all possible dummy variable sto 0
    v4_l= v4_u= v4_y= v6_W= v6_aa= v6_c= v6_cc= \
    v6_d= v6_e= v6_ff= v6_i= v6_j= v6_k= v6_m= v6_q=\
    v6_r= v6_x= v7_bb= v7_dd= v7_ff= v7_h= v7_j= v7_n= \
    v7_o= v7_v= v7_z= v9_f= v9_t= v10_f= v10_t= v12_f= \
    v12_t= v13_g= v13_p= v13_s   =0
    
    # Dynamically assign dummy variables
    globals()['v4'+'_'+obs['v4']] = 1
    globals()['v6'+'_'+obs['v6']] = 1
    globals()['v7'+'_'+obs['v7']] = 1
    globals()['v9'+'_'+obs['v9']] = 1
    globals()['v12'+'_'+obs['v12']] = 1
    globals()['v13'+'_'+obs['v13']] = 1
    
    # LR Equation obtained finallythrough modelling     
    predicted_prob = (df_1.v81*-0.123644269861552)+ (df_1.v11*-0.447073339958926)\
    + (df_1.log_v21*-0.0288012307212634)+ (df_1.v22_flag*-0.596490435792475)\
    + (df_1.log_v82*-0.0417904925302058)+ (df_1.log_v14*-0.102873625285458)\
    + (df_1.log_v15*-0.0757667215573646)+ (v4_l*-2.88117212656897)+ (v4_u*1.3154307113887)\
    + (v4_y*1.7007720481958)+ (v6_W*-1.60257943583471)+ (v6_aa*0.0273668415209801)\
    + (v6_c*-1.58752721258582)+ (v6_cc*-2.81537110199068)+ (v6_d*-1.71313054497753)\
    + (v6_e*-0.891269715194276)+ (v6_ff*4.13685945383271)+ (v6_i*0.95225657978722)\
    + (v6_j*6.2014549832735)+ (v6_k*0.660509209474622)+ (v6_m*-0.41001264496956)\
    + (v6_q*-0.960446940119353)+ (v6_r*1.48974925762289)+ (v6_x*-3.35282809683141)\
    + (v7_bb*-1.55966662146935)+ (v7_dd*1.89723131908799)+ (v7_ff*-0.696488205498239)\
    + (v7_h*1.55841199889135)+ (v7_j*-4.17037302863423)+ (v7_n*-2.1200046748122)\
    + (v7_o*3.56115987282087)+ (v7_v*1.23003329158014)+ (v7_z*0.434726681053774)\
    + (v9_f*2.39340916824807)+ (v9_t*-2.25837853523251)+ (v10_f*-0.694960908215702)\
    + (v10_t*0.829991541233886)+ (v12_f*0.213009963078659)+ (v12_t*-0.0779793300566063)\
    + (v13_g*-1.8654377015051)+ (v13_p*2.75859369316991)+ (v13_s*-0.758125358663091) \
    + 0.13503063
    
    # threshold is fixed at 0.7
    if predicted_prob.values > 0.7:
        predicted_value = 1
    else:
        predicted_value = 0
        
    return predicted_value
    
@app.route('/predict', methods=['POST'])
def get_prediction():
    
    if request.method == 'POST':
     obs = request.get_json()
     
     for key, value in obs.items():
         if not value:
             return jsonify({'error': 'value for {} is empty'.format(key)})
     prediction = model(obs)
     print(obs)
     print(prediction)
     return jsonify(prediction)
	    # Render template
	 
		## TODO
		# Modelling
		# LR Equation
		# Prediction
		


@app.route('/update', methods = ['POST','GET'])
def make_update():

	if request.method == 'POST':
		
		obs = request.get_json()
		### Update Equation ###
		## the weight should be declared globally ##


if __name__ == '__main__':
	app.run(debug=True)