import numpy as np
import pickle
import warnings

# Suppress the version mismatch warning
warnings.filterwarnings("ignore", category=UserWarning, module='sklearn')

# Load the pre-trained model
loaded_model = pickle.load(open("C:/Users/DELL/Downloads/Projects/ML Projects/Diabetes Prediction/trained_model_diabetes.sav", 'rb'))

input_data =(1,85,66,29,0,26.6,0.351,31)

#change the input_data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

#reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = loaded_model.predict(input_data_reshaped)
print(prediction)

if prediction[0]==0:
  print("Not diabetic")

else:
  print("Diabetic")