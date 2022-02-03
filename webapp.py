#import statements
import streamlit as st
import pandas as pd
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import plotly.figure_factory as ff
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import seaborn as sns
from PIL import Image



df = pd.read_csv(r'data.csv')

#titles
st.sidebar.header('Patient Data')
st.subheader('Training Dataset')
st.write(df.describe())


#train data.
x = df.drop(['Outcome'], axis = 1)
y = df.iloc[:, -1]
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state = 0)


#User reports
def user_report():
  Age = st.sidebar.slider('Age', 0,100, 54)
  Radius = st.sidebar.slider('Radius', 0,30, 15 )
  Texture = st.sidebar.slider('Texture', 0,40, 20 )
  Perimeter = st.sidebar.slider('Perimeter', 40,200, 92 )
  Area = st.sidebar.slider('Area', 140,2600, 650 )
  Smoothness = st.sidebar.slider('Smoothness', 0.0,0.25, 0.1 )
  Compactness = st.sidebar.slider('Compactness', 0.0,0.4, 0.1 )
  Concavity = st.sidebar.slider('Concavity', 0.0,0.5, 0.1 )
  Concave_points = st.sidebar.slider('Concave points', 0.0,0.25, 0.05 )
  Symmetry = st.sidebar.slider('Symmetry', 0.0,0.4, 0.2 )
  Fractal_Dimension = st.sidebar.slider('Fractal Dimension', 0.0,0.1, 0.06 )
  
  
  
  
  
  user_report_data = {
      'Age':Age,
      'Radius':Radius,
      'Texture':Texture,
      'Perimeter':Perimeter,
      'Area':Area,
      'Smoothness':Smoothness,
      'Compactness':Compactness,
      'Concavity':Concavity,
      'Concave_points':Concave_points,
      'Symmetry':Symmetry,
      'Fractal_Dimension':Fractal_Dimension,
      
        
  }
  report_data = pd.DataFrame(user_report_data, index=[0])
  return report_data





user_data = user_report()
st.subheader('Patient Data')
st.write(user_data)





rf  = RandomForestClassifier()
rf.fit(x_train, y_train)
user_result = rf.predict(user_data)



#Visualization
st.title('Graphical Patient Report')



if user_result[0]==0:
  color = 'green'
else:
  color = 'red'


st.header('Radius Value Graph (Yours vs Others)')
fig_Radius = plt.figure()
ax3 = sns.scatterplot(x = 'Age', y = 'Radius', data = df, hue = 'Outcome' , palette='Purples')
ax4 = sns.scatterplot(x = user_data['Age'], y = user_data['Radius'], s = 150, color = color)
plt.xticks(np.arange(0,100,5))
plt.yticks(np.arange(0,50,5))
plt.title('0 - Healthy & 1 - Unhealthy')
st.pyplot(fig_Radius)



st.header('Texture Value Graph (Yours vs Others)')
fig_Texture = plt.figure()
ax9 = sns.scatterplot(x = 'Age', y = 'Texture', data = df, hue = 'Outcome', palette='rainbow')
ax10 = sns.scatterplot(x = user_data['Age'], y = user_data['Texture'], s = 150, color = color)
plt.xticks(np.arange(0,100,5))
plt.yticks(np.arange(0,50,5))
plt.title('0 - Healthy & 1 - Unhealthy')
st.pyplot(fig_Texture)



st.header('Perimeter Value Graph (Yours vs Others)')
fig_Perimeter = plt.figure()
ax5 = sns.scatterplot(x = 'Age', y = 'Perimeter', data = df, hue = 'Outcome', palette='Blues')
ax6 = sns.scatterplot(x = user_data['Age'], y = user_data['Perimeter'], s = 150, color = color)
plt.xticks(np.arange(0,100,5))
plt.yticks(np.arange(0,200,10))
plt.title('0 - Healthy & 1 - Unhealthy')
st.pyplot(fig_Perimeter)



st.header('Area Value Graph (Yours vs Others)')
fig_Area = plt.figure()
ax11 = sns.scatterplot(x = 'Age', y = 'Area', data = df, hue = 'Outcome', palette='Greens')
ax12 = sns.scatterplot(x = user_data['Age'], y = user_data['Area'], s = 150, color = color)
plt.xticks(np.arange(0,100,5))
plt.yticks(np.arange(100,2500,100))
plt.title('0 - Healthy & 1 - Unhealthy')
st.pyplot(fig_Area)


st.header('Smoothness Value Graph (Yours vs Others)')
fig_Smoothness = plt.figure()
ax13 = sns.scatterplot(x = 'Age', y = 'Smoothness', data = df, hue = 'Outcome', palette='rocket')
ax14 = sns.scatterplot(x = user_data['Age'], y = user_data['Smoothness'], s = 150, color = color)
plt.xticks(np.arange(0,100,5))
plt.yticks(np.arange(0.0,0.25,0.0125))
plt.title('0 - Healthy & 1 - Unhealthy')
st.pyplot(fig_Smoothness)


st.header('Compactness count Graph (Yours vs Others)')
fig_Compactness = plt.figure()
ax1 = sns.scatterplot(x = 'Age', y = 'Compactness', data = df, hue = 'Outcome', palette = 'magma')
ax2 = sns.scatterplot(x = user_data['Age'], y = user_data['Compactness'], s = 150, color = color)
plt.xticks(np.arange(0,100,5))
plt.yticks(np.arange(0.0,0.4,0.02))
plt.title('0 - Healthy & 1 - Unhealthy')
st.pyplot(fig_Compactness)



st.header('Concavity Value Graph (Yours vs Others)')
fig_Concavity = plt.figure()
ax7 = sns.scatterplot(x = 'Age', y = 'Concavity', data = df, hue = 'Outcome', palette='Reds')
ax8 = sns.scatterplot(x = user_data['Age'], y = user_data['Concavity'], s = 150, color = color)
plt.xticks(np.arange(0,100,5))
plt.yticks(np.arange(0.0,0.5,0.025))
plt.title('0 - Healthy & 1 - Unhealthy')
st.pyplot(fig_Concavity)



st.header('Concave points Value Graph (Yours vs Others)')
fig_Concavepoints = plt.figure()
ax7 = sns.scatterplot(x = 'Age', y = 'Concave points', data = df, hue = 'Outcome', palette='mako')
ax8 = sns.scatterplot(x = user_data['Age'], y = user_data['Concave_points'], s = 150, color = color)
plt.xticks(np.arange(0,100,5))
plt.yticks(np.arange(0.0,0.25,0.0125))
plt.title('0 - Healthy & 1 - Unhealthy')
st.pyplot(fig_Concavepoints)




st.header('Symmetry Value Graph (Yours vs Others)')
fig_Symmetry = plt.figure()
ax7 = sns.scatterplot(x = 'Age', y = 'Symmetry', data = df, hue = 'Outcome', palette='flare')
ax8 = sns.scatterplot(x = user_data['Age'], y = user_data['Symmetry'], s = 150, color = color)
plt.xticks(np.arange(0,100,5))
plt.yticks(np.arange(0.0,0.4,0.02))
plt.title('0 - Healthy & 1 - Unhealthy')
st.pyplot(fig_Symmetry)


st.header('Fractal Dimension Value Graph (Yours vs Others)')
fig_FractalDimension = plt.figure()
ax7 = sns.scatterplot(x = 'Age', y = 'Fractal Dimension', data = df, hue = 'Outcome', palette='crest')
ax8 = sns.scatterplot(x = user_data['Age'], y = user_data['Fractal_Dimension'], s = 150, color = color)
plt.xticks(np.arange(0,100,5))
plt.yticks(np.arange(0.0,0.1,0.005))
plt.title('0 - Healthy & 1 - Unhealthy')
st.pyplot(fig_FractalDimension)




st.subheader('Your Report: ')
output=''
if user_result[0]==0:
  output = 'Congratulations, you do not have  Breast Cancer'
else:
  output = 'Unfortunately, you do have Breast Cancer'
st.title(output)

st.subheader('Accuracy: ')
st.write(str(accuracy_score(y_test, rf.predict(x_test))*100)+'%')

