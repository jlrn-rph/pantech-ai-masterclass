# MACHINE LEARNING ALGORITHMS ON DIABETES DATASET - TRAINING THE MODEL

# import libraries
import pandas as pd 
from pandas.plotting import scatter_matrix
import pickle
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold

# load dataset
url = 'machine-learning/ml-algorithm/dataset/diabetes.csv'
cols = ['Pregnancies', 'Glucose', 'BloodPressure','SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
dataframe = pd.read_csv(url)

# summarize dataframe
print(dataframe.head(5)) 
print(dataframe.shape) 
print(dataframe.describe) 
print(dataframe.groupby('Outcome').size())

# split dataframe's features and labels
array = dataframe.values
X = array[:, 0:8]
y = array[:, 8]
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.20, random_state = 42, shuffle = True)

# initialize algorithms
models = []
models.append(('Logistic Regression', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('Linear Discriminant Analysis', LinearDiscriminantAnalysis()))
models.append(('K Nearest Neighbors', KNeighborsClassifier()))
models.append(('Classification and Regression Tree', DecisionTreeClassifier()))
models.append(('Gaussian NB', GaussianNB()))
models.append(('Support Vector Machine', SVC(gamma='auto')))

# evaluate the algorithms
results = []
cols = []
res = []
for col, model in models:
    kfold = StratifiedKFold(n_splits=10, random_state=None)
    cv_results = cross_val_score(model, X_train, y_train, cv = kfold, scoring = 'accuracy')
    results.append(cv_results)
    cols.append(col)
    res.append(cv_results.mean())
    print('%s: %f (%f)' % (col, cv_results.mean(), cv_results.std()))

plt.bar(cols, res, color ='maroon', width = 0.6)

plt.title('Algorithm Comparison')
plt.show()

# training the model
model_lr = LogisticRegression(solver='liblinear', multi_class='ovr') # initialize the model

model_lr.fit(X_train, y_train)  # train the model

# save the model
filename = 'machine-learning/ml-algorithm/model/model.pkl'
pickle.dump(model_lr, open(filename, 'wb'))
print('Model saved')

# check validation metrics
loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.score(X_val, y_val)
print('Validation accuracy:', result)