import pickle
import pandas as pd
import numpy as np
from unittest import result
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt

st.title('Prognostic Models in Critically Ill Patients with Sepsis-associated Acute Kidney Injury')
st.subheader('Application of real-time prediction of single new sample')
st.subheader('View some data of training cohort')
##展示原始数据部分
disp=pd.read_csv('E:/MIMIC/AKI-spesis/streamlit/1882-subjectid.csv')
st.dataframe(disp,1000,300 )


no_features = ['status_28d','status_14d', 'status_7d', 'survival_time']

# 数据文件
data=  pd.read_csv("E:\MIMIC\AKI-spesis\streamlit\X_feature71428.csv")
features = list(data.columns)
for feature in no_features:
	if feature in features:
		features.remove(feature)


r = pd.read_excel("E:\MIMIC\AKI-spesis\streamlit\变量范围.xlsx",index_col=0).fillna('').T
dic = {}
for index,row in r.iterrows():
	dic[index] = row['最大值'],row['最小值'],row['变量类型'],row['单位']

import os
import streamlit as st
st.sidebar.header("Prediction system")
# 模型存储路径
model_folder = 'E:\MIMIC\AKI-spesis\streamlit'

target = st.sidebar.selectbox("Choose your target",["status_7d",'status_14d','status_28d'])
models = [i for i in os.listdir(model_folder) if i.endswith('.pkl') and 'trained_model' in i and target in i]
model_dic = {model.split('_trained_model_')[0]:model for model in models}
if len(models)==0:
	st.sidebar.warning(f"No models found for {target}")
model_name = st.sidebar.selectbox("Choose the model",model_dic.keys())
values = {}
for feature in features:
	a,b,c,d = dic[feature]
	v  =st.sidebar.slider(label= feature + f'({d})' if d!='' else feature+ '',
						min_value= float(b) if c !='分类' else int(b),
						max_value= float(a) if c !='分类' else int(a),
						step = 0.01 if c !='分类' else 1,
						)

	values[feature] = float(v)

start = st.sidebar.button("Start")
if start:
	model_path = os.path.join(model_folder,
							model_dic[model_name])
	with open(model_path,'rb') as f:
		model = pickle.load(f)
	X = np.array([list(values.values())])
	st.markdown("#### The values you entered are：")
	st.write(values)
	y_pred = model.predict(X)
	y_pred_prob = model.predict_proba(X)
	st.success(f"The prediction is successful. The result of {target} is")
	res = y_pred[0]
	res = 'Survival' if res else 'Nonsurvival'
	st.markdown(f"#### {res}")
	st.success("Probability is：")
	st.table([['Survival','Nonsurvival'],[y_pred_prob[0][0],y_pred_prob[0][1]]])


st.text('In the left sidebar, you can select the outcome and model to be predicted.')
st.text('Here, we recommend RF model and GBDT model for prediction.')
st.text('Among the variables listed, SOFA represents the score of patients at ICU admission.')
st.text('The remaining numerical variables were values during the first 24 hours of ICU admission. ')
st.text('In the categorical variables, AKI_stage 0, 1, 2 represents AKI stage 1, 2, 3, respectively. ')
st.text('For the remaining categorical variables, 0 represents no disease; 1 means having the disease;')