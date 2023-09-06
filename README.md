# BrainStrokeClassifier API

This repo contains the source code for the Python-based Brain Stoke Diagnostic package. The package is published on Python's open-source platform [PyPI](https://pypi.org/project/BrainStrokeClassifier/). It takes a couple of data points regarding the patient's health and uses a pre-trained state-of-the-art machine learning model to diagnose whether the patient has a brain stroke or not.

[![Python 3.9](https://img.shields.io/badge/python-3.6-blue.svg)](https://www.python.org/downloads/release/python-360/)
# Functionality of the BrainStrokeClassifier

The package includes a trained Machine Learning model to predict Brain Stroke
vulnerability. The model can be accessed as **BStClassifier** utility. You can use **BStClassifier(X)** to make predictions. Where **X** is the input consisting of 7 characteristics:

1. Gender ("Male"=0, "Female"=1, "Other"=-1)
2. Age
3. Do you have hypertension (Yes=1/No=0)
4. Do you have any heart disease (Yes=1/No=0)
5. What type of work you do? ("Private job"=0, "Self Employed"=1, "Govt. Job"=2, "Never Worked"=-2, "You are a Child"=-1)
6. Your average blood glucose level
7. Your Body Mass Index (BMI)

It classifies whether you are vulnerable to or had brain stroke, or not.
 
## Input type

The input can be:
1. NumPy array

         a. 1D array of shape (7,)
         
         b. 2D array of shape (S,7); where S is number of samples

2. Pandas Series of shape (7,)
3. Pandas DataFrame of shape (S,7); where S is number of samples

# About the Machine Learning model

The Machine Learning model is a Pipeline of Standard Scaler and XGBoost Classifier. You can check the notebook where the model is trained through the [github](https://github.com/MUmairAB/Stroke-Prediction-using-Machine-Learning/blob/a0c9126b4f8c55ea528b1f60354f441148173567/stroke-prediction.ipynb)

# Package Installation

Make sure you have Python installed in your system. Then, run the following command in the command window.
 ```
  !pip install BrainStrokeClassifier
  ```

# Example Code

 ```
# test.py


# Package installation
!pip install BrainStrokeClassifier

# Loading package
from BrainStrokeClassifier import BStClassifier
import numpy as np
import pandas as pd

# Creating some test input

x = np.array([[0,67,0,1,0,228.69,36.6],
              [0,58,1,0,0,87.96,39.2]])

df = pd.DataFrame(x)
prediction = BStClassifier(df)
print(prediction)
 ```
# For Suggestions

**If you have any suggestions or improvements, please fork the package on [issues](https://github.com/MUmairAB/BrainStrokeClassifier/issues)**

# Credits

**The Umair Akram | MUmairAB**
