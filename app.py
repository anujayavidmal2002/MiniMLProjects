import numpy as np
import pickle
import streamlit as st


# Load the pre-trained model
loaded_model = pickle.load(open("C:/Users/DELL/Downloads/Projects/ML Projects/Diabetes Prediction/trained_model_diabetes.sav", 'rb'))

# creating a function fro prediction
def diabetes_prediction(input_data):
 #input_data =(1,85,66,29,0,26.6,0.351,31)

 #change the input_data to numpy array
 input_data_as_numpy_array = np.asarray(input_data)

 #reshape the array as we are predicting for one instance
 input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

 prediction = loaded_model.predict(input_data_reshaped)
 print(prediction)

 if prediction[0]==0:
   return "Not diabetic detected"

 else:
   return "Diabetic detected"
 

def main():
  
 #giving a title 
 st.title('DIabetes Prediction Web App')
  
 #getting the input data from the user
 Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age

Pregnancies = st.text_input("Number of Pregnancies")
Glucose = st.text_input("Glucose Level")
BloodPressure = st.text_input("Blood Pressure Values")
SkinThickness = st.text_input("Skin Thickness Value")
Insulin = st.text_input("Insulin Level")
BMI = st.text_input("BMI Value")
DiabetesPedigreeFunction = st.text_input("Diabetes Pedigree Function value")
Age = st.text_input("Age of the person")

#code for prediction 
diagnosis = ''
# creating a button for prediction
if st.button("Diabetes Test Results"):
  diagnosis = diabetes_prediction([Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age])

st.success(diagnosis)



if __name__ =='__main__':
  main()