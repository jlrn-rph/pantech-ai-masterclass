# MACHINE LEARNING ALGORITHMS ON DIABETES DATASET - TESTING THE MODEL

# import libraries
import pandas as pd 
import pickle

# load model
file = 'machine-learning/ml-algorithm/model/model.pkl'
load_model = pickle.load(open(file, 'rb'))

# get user input
preg = float(input('Pregnancy: '))
glu = float(input('Glucose: '))
bp = float(input('Blood Pressure: '))
skin = float(input('Skin: '))
ins = float(input('Insulin: '))
bmi = float(input('BMI: '))
ped = float(input('Diabetes Pedigree Function: '))
age = float(input('Age: '))

# predict user input 
value = [[preg, glu, bp, skin, ins, bmi, ped, age]]
preds = load_model.predict(value)
print('Predicted value:', preds[0])