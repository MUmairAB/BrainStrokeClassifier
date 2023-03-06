# -*- coding: utf-8 -*-
"""
@author: Umair Akram

The package includes a trained Machine Learning model to predict Brain Stroke
vulnerability. It takes 7 characteristics:
    1. Gender ("Male", "Female", "Other")
    2. Age
    3. Do you have hypertension (Yes/NO)
    4. Do you have any heart disease (Yes/NO)
    5. What type of work you do? ("Private job", "Self Employed", "Govt. Job", "Never Worked", "You are a Child")
    6. Your average blood glucose level
    7. Your Body Mass Index (BMI)
and classifies whether you are vulnerable to or had brain stroke, or not.
 
the input can be:
    1. NumPy array
         a. 1D array of shape (7,)
         b. 2D array of shape (S,7); where S is number of samples
    2. Pandas Series of shape (7,)
    3. Pandas DataFrame of shape (S,7); where S is number of columns
    


Use Case:

# Package installation
!pip install BrainStrokeClassifier

# Loading package
from BrainStrokeClassifier import BStClassifier

x = np.array([[0,67,0,1,0,228.69,36.6],
              [0,58,1,0,0,87.96,39.2]])

df = pd.DataFrame(x)
prediction = BStClassifier(df)
print(prediction)


"""
import numpy as np
import pandas as pd
import pickle


#######        Check datatype of input        #############
def check_dtype(x_input):
    
    if isinstance(x_input,pd.core.series.Series):
        return 'Series'
    elif isinstance(x_input,pd.core.frame.DataFrame):
        return 'DataFrame'
    elif isinstance(x_input,np.ndarray):
        return 'NumPy array'
    else:
        raise Exception('Bad Input. Iput can only be a list or NumPy array (of length 7), Pandas Series (of length 7) or DataFrame (with 7 columns)')


#######     Converting user input to DataFrame     ############
def input_manipulation(user_input):
    d_type = check_dtype(user_input)
    features = ['gender','age','hypertension','heart_disease','work_type','avg_glucose_level','bmi']
    # default return value
    val = [0,67,0,1,0,228.69,36.6]
        
    ###################    NumPy array input   ################
    if d_type == 'NumPy array':
        
        # 1D NumPy array
        if user_input.ndim == 1:
            if user_input.shape[0] != 7:
                raise Exception('The 1D array must have the shape (7,)')
        
            # converting 1D NumPy array to a DataFrame
            val = pd.DataFrame(data=user_input.reshape(1,-1),
                           columns=features)
            return val
    
        # 2D NumPy array
        elif user_input.ndim == 2:
            if user_input.shape[1] != 7:
                raise Exception('The 2D array must have 7 columns')
        
            # converting 2D NumPy array to a DataFrame
            val = pd.DataFrame(data=user_input,
                           columns=features)
            return val
        
        else:
            raise Exception('NumPy array dimension cannot exceed than 2D')
    
    
    ###################    Series input   ################
    elif d_type == 'Series':
        if user_input.shape[0] != 7:
            raise Exception('The Pandas Series must have the shape (7,)')
        
        # converting Series to a DataFrame
        val = pd.DataFrame(data=user_input.to_numpy().reshape(1,-1),
                           columns=features)
        return val
        
    
    ###################    DataFrame   ################
    elif d_type == 'DataFrame':
        if user_input.shape[1] != 7:
            raise Exception('The DataFrame must have 7 columns')
        
        # manipulation the DataFrame
        val = pd.DataFrame(data=user_input.values,
                           columns=features)
        return val
    else:
        raise Exception('Bad input')
    
    
##############     Making predictions     #####################
def BStClassifier(input1):
        
    # Converting input to a Pandas DataFrame
    input_df = input_manipulation(input1)
    
    # Loading pre-trained model
    model = pickle.load(open('trained_model.sav','rb'))
    
    predictions = model.predict(input_df)
    return predictions
