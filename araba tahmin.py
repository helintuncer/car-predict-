import pandas as pd
df = pd.read_csv('cars.csv')

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

x = df.drop('Price',axis=1)
y =df[['Price']]


x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=41)

preproccer=ColumnTransformer(transformers=[('num',StandardScaler(),
                                           ['Mileage','Cylinder','Liter','Doors']),
                            ('cat',OneHotEncoder(),['Make','Model','Trim','Type'])])


model = LinearRegression()
pipe = Pipeline(steps=[('preproccer', preproccer),('model',model)])
pipe.fit(x_train, y_train)
y_pred = pipe.predict(x_test)

import streamlit as st
from sklearn import pipeline


def price(make,model,trim,mileage,car_type,cylinder,liter,doors,cruise,sound,leather):
	input_data=pd.DataFrame({
		'Make':[make],
		'Model':[model],
		'Trim':[trim],
		'Mileage':[mileage],
		'Type':[car_type],
		'Car_type':[car_type],
		'Cylinder':[cylinder],
		'Liter':[liter],
		'Doors':[doors],
		'Cruise':[cruise],
		'Sound':[sound],
		'Leather':[leather]
		})
	prediction=pipe.predict(input_data)[0]



	return prediction
st.title("Car Price Prediction :red_car: @drmurataltun")
st.write("Enter Car Details to predict the price of the car")
make=st.selectbox("Make",df['Make'].unique())
model=st.selectbox("Model",df[df['Make']==make]['Model'].unique())
trim=st.selectbox("Trim",df[(df['Make']==make) & (df['Model']==model)]['Trim'].unique())
mileage=st.number_input("Mileage",200,60000)
car_type=st.selectbox("Type",df['Type'].unique())
cylinder=st.selectbox("Cylinder",df['Cylinder'].unique())
liter=st.number_input("Liter",1,6)
doors=st.selectbox("Doors",df['Doors'].unique())
cruise=st.radio("Cruise",[True,False])
sound=st.radio("Sound",[True,False])
leather=st.radio("Leather",[True,False])
if st.button("Predict"):
	pred=price(make,model,trim,mileage,car_type,cylinder,liter,doors,cruise,sound,leather)

	st.write("Predicted Price :red_car:  $",round(pred[0],2))