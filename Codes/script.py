import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import openpyxl
import graphviz
import PySimpleGUI as sg
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow import keras
from keras.layers import Dense
from keras.models import Sequential
from ann_visualizer.visualize import ann_viz
from sklearn.metrics import mean_squared_error
from graphviz import Source

def ANN(neurons):
 #Import data
 data = pd.read_excel('car_sales_data.xlsx', engine='openpyxl')


 #Create input dataset from data
 x = data.drop(['Customer Name', 'Customer e-mail', 'Country', 'Car Purchase Amount'], axis = 1)
 #Show Input Data
 print(x) 
 #Show Input Shape
 print("Input data Shape=",x.shape)

 #Create output dataset from data
 y = data['Car Purchase Amount']
 #Show Output Data
 print(y)
 #Transform Output
 y = y.values.reshape(-1,1)
 #Show Output Transformed Shape
 print("Output Data Shape=",y.shape)

 #Scale input
 scaler_in = MinMaxScaler()
 x_scaled = scaler_in.fit_transform(x)
 print(x_scaled)

 #Scale output
 scaler_out = MinMaxScaler()
 y_scaled = scaler_out.fit_transform(y)
 print(y_scaled)

 x_train, x_test, y_train, y_test = train_test_split(x_scaled, y_scaled, test_size=0.2)
 neurons = 20

 #Create model
 model = Sequential()
 model.add(Dense(neurons, input_dim=5, activation='relu'))
 model.add(Dense(neurons, activation='relu'))
 model.add(Dense(1, activation='linear'))
 print(model.summary())

 #Train model
 model.compile(optimizer= 'adam', loss = 'mean_squared_error')
 epochs_hist = model.fit(x_train, y_train, epochs=10, batch_size=10, verbose=1, validation_data=(x_test, y_test))
 print(epochs_hist.history.keys()) #print dictionary keys

 """
# User Input Code + Plotting - Commented for testing purposes - remove triple " to execute this
 # Colour theme to input window
 sg.theme('SandyBeach')     
  
 # Input window from PySimpleGUI
 layout = [
    [sg.Text('Please enter your information below')],
    [sg.Text('Gender 0 (M), 1 (F)', size =(15, 1)), sg.InputText()],
    [sg.Text('Age', size =(15, 1)), sg.InputText()],
    [sg.Text('Annual Salary', size =(15, 1)), sg.InputText()],
    [sg.Text('Credit Card Debt', size =(15, 1)), sg.InputText()],
    [sg.Text('Net Worth', size =(15, 1)), sg.InputText()],
    [sg.Submit(), sg.Cancel()]
 ]
  
 window = sg.Window('Predict purchase amount', layout)
 event, values = window.read()
 window.close()
  
 # Print values
 print(event, values[0], values[1], values[2], values[3], values[4])

 # Evaluate model based on user input
 input_user_data = np.array([[values[0], values[1], values[2], values[3], values[4]]])

 #Scale user input data
 input_user_data_scaled = scaler_in.transform(input_user_data)

 #Predict output
 output_predict_data_scaled = model.predict(input_user_data_scaled)

 #Print predicted output
 print('Predicted Output (Scaled) =', output_predict_data_scaled)

 #Unscale output
 output_predict_sample = scaler_out.inverse_transform(output_predict_data_scaled)
 print('Predicted Output / Purchase Amount ', output_predict_sample)

  #Plot the training graph to see how quickly the model learns
 plt.plot(epochs_hist.history['loss'])
 plt.plot(epochs_hist.history['val_loss'])
 plt.title('Model Loss Progression During Training/Validation')
 plt.ylabel('Training and Validation Losses')
 plt.xlabel('Epoch Number')
 plt.legend(['Training Loss', 'Validation Loss'])
 plt.show()
 
 #Code to create and visualize pictorial view of the model
 #ann_viz(model, filename="modelview.gv")
 s = Source.from_file('modelview.gv')
 s.view()
 """

 predictions = model.predict(x_test)
 MSE = mean_squared_error (y_test, predictions)
 print ('Error %: ', MSE)

 return MSE
#ANN(20)  #remove # to execute code without testing / add # to test and execute pytest



""" #Test data for SR6 - Deprecated to implement user input

# Evaluate model based on test sample data
# Gender, Age, Annual Salary, Credit Card Debt, Net Worth 
​
input_test_sample = np.array([[0, 41.8,  62812.09, 11609.38, 238961.25]])
​
# Scale input test sample data
input_test_sample_scaled = scaler_in.transform(input_test_sample)
​
# Predict output
output_predict_sample_scaled = model.predict(input_test_sample_scaled)
​
# Print predicted output
print('Predicted Output (Scaled) =', output_predict_sample_scaled)
​
# Unscale output
output_predict_sample = scaler_out.inverse_transform(output_predict_sample_scaled)
print('Predicted Output / Purchase Amount ', output_predict_sample)
"""





