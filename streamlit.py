import requests
import json
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import plotly.figure_factory as ff
import matplotlib.pyplot as plt
import shap

st.title('Credit dashboard')
# st.write("""# Explore different variables""")
st.write("""Explore different variables""")

train_df = pd.read_csv('creditdf.csv')

clients = train_df.head(100)
clients = clients[['Client']]

selected_status = st.selectbox(
    'Select variable',
    options=['AGE', 'AMT_CREDIT', 'AMT_INCOME_TOTAL','DAYS_EMPLOYED'])

client_number = st.sidebar.selectbox(
    'Select Client',
    (clients))

st.sidebar.write('You selected: ', client_number)

rslt_df = train_df[(train_df['Client'] == client_number)]
gender=rslt_df[['CODE_GENDER_F']] == 0
gender=np.array(gender)
education=rslt_df[['Higher_Education']]==0
education=np.array(education)
working=rslt_df[['Working']]==0
working=np.array(working)

if gender == True:
    st.sidebar.write("Gender: Male")
if gender == False:
    st.sidebar.write("Gender: Female")

if education == True:
    st.sidebar.write("Higher education: No")
if education == False:
    st.sidebar.write("Higher education: Yes")

if working == True:
    st.sidebar.write("Income type: Other")
if working == False:
    st.sidebar.write("Income type: Working")

bday1=np.mean(train_df[train_df["TARGET"] == 1])
bday1=bday1['AGE']
bday2=np.mean(train_df[train_df["TARGET"] == 0])
bday2=bday2['AGE']
cbday=float(rslt_df['AGE'])

if selected_status == 'AGE':
    st.write('Candidate: ', cbday)
    st.write('Non-Falter mean: ', bday2)
    st.write('Falter mean: ', bday1)

credit1=np.mean(train_df[train_df["TARGET"] == 1])
credit1=credit1['AMT_CREDIT']
credit2=np.mean(train_df[train_df["TARGET"] == 0])
credit2=credit2['AMT_CREDIT']
ccredit=float(rslt_df['AMT_CREDIT'])

if selected_status == 'AMT_CREDIT':
    st.write('Candidate: ', ccredit)
    st.write('Non-Falter mean: ', credit2)
    st.write('Falter mean: ', credit1)

income1=np.mean(train_df[train_df["TARGET"] == 1])
income1=income1['AMT_INCOME_TOTAL']
income2=np.mean(train_df[train_df["TARGET"] == 0])
income2=income2['AMT_INCOME_TOTAL']
cincome=float(rslt_df['AMT_INCOME_TOTAL'])

if selected_status == 'AMT_INCOME_TOTAL':
    st.write('Candidate: ', cincome)
    st.write('Non-Falter mean: ', income2)
    st.write('Falter mean: ', income1)

employed1=np.mean(train_df[train_df["TARGET"] == 1])
employed1=employed1['DAYS_EMPLOYED']
employed2=np.mean(train_df[train_df["TARGET"] == 0])
employed2=employed2['DAYS_EMPLOYED']
cemployed=float(rslt_df['DAYS_EMPLOYED'])

if selected_status == 'DAYS_EMPLOYED':
    st.write('Candidate: ', cemployed)
    st.write('Non-Falter mean: ', employed2)
    st.write('Falter mean: ', employed1)

if selected_status == 'AGE':
    hist_data = [train_df[train_df["TARGET"] == 1]["AGE"], train_df[train_df["TARGET"] == 0]["AGE"]]
    group_labels = ["Falter", "Non-Falter"]
    fig = ff.create_distplot(hist_data, group_labels,show_hist=False)
    fig['layout'].update(title={"text": 'Distribution of clients'}, xaxis_title="Days of Birth", yaxis_title="density")
    fig.add_vline(x=cbday, line_width=1, line_color="black")
    st.plotly_chart(fig, use_container_width=True)

if selected_status == 'AMT_CREDIT':
    hist_data = [train_df[train_df["TARGET"] == 1]["AMT_CREDIT"], train_df[train_df["TARGET"] == 0]["AMT_CREDIT"]]
    group_labels = ["Falter", "Non-Falter"]
    fig = ff.create_distplot(hist_data, group_labels,show_hist=False)
    fig['layout'].update(title={"text": 'Distribution of clients'}, xaxis_title="Amount of credit", yaxis_title="density")
    fig.add_vline(x=ccredit, line_width=1, line_color="black")
    st.plotly_chart(fig, use_container_width=True)

if selected_status == 'AMT_INCOME_TOTAL':
    hist_data = [train_df[train_df["TARGET"] == 1]["AMT_INCOME_TOTAL"], train_df[train_df["TARGET"] == 0]["AMT_INCOME_TOTAL"]]
    group_labels = ["Falter", "Non-Falter"]
    fig = ff.create_distplot(hist_data, group_labels,show_hist=False)
    fig['layout'].update(title={"text": 'Distribution of clients'}, xaxis_title="Income", yaxis_title="density")
    fig.add_vline(x=cincome, line_width=1, line_color="black")
    st.plotly_chart(fig, use_container_width=True)

if selected_status == 'DAYS_EMPLOYED':
    hist_data = [train_df[train_df["TARGET"] == 1]["DAYS_EMPLOYED"], train_df[train_df["TARGET"] == 0]["DAYS_EMPLOYED"]]
    group_labels = ["Falter", "Non-Falter"]
    fig = ff.create_distplot(hist_data, group_labels,show_hist=False)
    fig['layout'].update(title={"text": 'Distribution of clients'}, xaxis_title="Days employed", yaxis_title="density")
    fig.add_vline(x=cemployed, line_width=1, line_color="black")
    st.plotly_chart(fig, use_container_width=True)

st.title('Prediction')
rn=int((train_df[train_df['Client']==client_number].index).values)
train_df=train_df.drop(['Unnamed: 0'],axis=1)
train_df=train_df.drop(['Client'],axis=1)
X,y = (train_df.drop(['TARGET'],axis=1).values,train_df.TARGET.values)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.33,random_state=0)
rf = RandomForestClassifier(max_depth=4 , random_state=0,class_weight='balanced')
model=rf.fit(X_train, y_train)

y_pred2 = rf.predict(X)
score = y_pred2[rn]
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

train_df=train_df.drop(['TARGET'],axis=1)

lcol=['Employment','Id publish','Region','Commuting','Source 2','Source 3','Phone','Gender','Credit','Income','Annuity','Working','Education','Age',]

if score==0:
    st.write('Predicted: Non-Falter')
if score==1:
    st.write('Predicted: Falter')

st.set_option('deprecation.showPyplotGlobalUse', False)
X_idx =rn
import streamlit.components.v1 as components
def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)

shap_value_single = explainer.shap_values(X = X[X_idx:X_idx+1,:])
plt.title('Feature importance based on SHAP values')
st_shap(shap.force_plot(base_value = explainer.expected_value[0],shap_values = shap_value_single[0],features = X[X_idx:X_idx+1,:],feature_names=lcol))
st.write('---')

st.title('New client details')

x1 = st.number_input('Age', min_value=0, max_value=100, value=0, step=1)
x2 = st.number_input('Days employed', min_value=0, max_value=100000, value=0, step=1)
x3 = st.number_input('Id Publish', min_value=0, max_value=100000, value=0, step=1)
x4 = st.number_input('Region', min_value=1, max_value=3, value=1, step=1)
option5=st.selectbox('Commuting',('Yes', 'No'))
if option5=='Yes':
  x5=1
if option5=='No':
  x5=0
#x5 = st.number_input('Commuting')
x6 = st.number_input('External source1', min_value=0, max_value=100000000, value=0)
x7 = st.number_input('External source2', min_value=0, max_value=100000000, value=0)
x8 = st.number_input('Last phone change', min_value=0, max_value=100000, value=0, step=1)
option9=st.selectbox('Gender',('Female', 'Male'))
if option9=='Female':
  x9=1
if option9=='Male':
  x9=0
#x8 = st.number_input('Gender')
option10=st.selectbox('Income type',('Working', 'Other'))
if option10=='Working':
  x10=1
if option10=='Other':
  x10=0
#x9 = st.number_input('Working')
option11=st.selectbox('Higher Education',('Yes', 'No'))
if option11=='Yes':
  x11=1
if option11=='No':
  x11=0
#x10 = st.number_input('Higher Education')
x12 = st.number_input('Amount of credit', min_value=0, max_value=10000000000, value=0, step=1)
x13 = st.number_input('Income', min_value=0, max_value=10000000000, value=0, step=1)
x14 = st.number_input('Annuity', min_value=0, max_value=10000000000, value=0, step=1)

#option = st.selectbox('How would you like to be contacted?',('Email', 'Home phone', 'Mobile phone'))

if st.button('Submit'):
  new_measurement = {
    'AGE': x1,
    'DAYS_EMPLOYED': x2,
    'DAYS_ID_PUBLISH': x3,
    "REGION_RATING_CLIENT": x4,
    "REG_CITY_NOT_WORK_CITY": x5,
    "EXT_SOURCE_2": x6,
    "EXT_SOURCE_3": x7,
    "DAYS_LAST_PHONE_CHANGE": x8,
    "CODE_GENDER_F": x9,
    "Working": x10,
    "Higher_Education": x11,
    "AMT_CREDIT": x12,
    "AMT_INCOME_TOTAL": x13,
    "AMT_ANNUITY": x14}
  response = requests.post('https://creditdashboard7.herokuapp.com/predict', json=new_measurement)
  #st.write(response.content)
  res_dict = json.loads(response.content.decode('utf-8'))
  #st.write(res_dict)
  result=int(res_dict.get("prediction"))
  prob =float(res_dict.get("probability"))

  if result==1:
    #st.write(result)
    st.write('Prediction:Falter')
    st.write('Probability', prob)
  if result==0:
    #st.write(result)
    st.write('Prediction: Non-Falter')
    st.write('Probability', prob)
  test14 = [[x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13,x14]]
  test14 = np.array(test14)
  shap_value_client = explainer.shap_values(X=test14 )
  plt.title('Feature importance based on SHAP values')
  st_shap(shap.force_plot(base_value=explainer.expected_value[0], shap_values=shap_value_single[0],
                          features=test14 , feature_names=lcol))

else:
  st.write('')
