import numpy as np
import pandas as pd


crop = pd.read_csv("Crop_recommendation .csv")
crop_dict = {
    'rice': 1,
    'maize': 2,
    'jute': 3,
    'cotton': 4,
    'coconut': 5,
    'papaya': 6,
    'orange': 7,
    'apple': 8,
    'muskmelon': 9,
    'watermelon': 10,
    'grapes': 11,
    'mango': 12,
    'banana': 13,
    'pomegranate': 14,
    'lentil': 15,
    'blackgram': 16,
    'mungbean': 17,
    'mothbeans': 18,
    'pigeonpeas': 19,
    'kidneybeans': 20,
    'chickpea': 21,
    'coffee': 22
}
crop['crop_num'] = crop['label'].map(crop_dict)
#print(crop)

crop.drop(['label'], axis=1, inplace=True)
#print(crop.head())

X = crop.drop(['crop_num'], axis=1)  #input
y = crop['crop_num']  #output
#print(X)
#print(y)

from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=50)
#print(X_train)
#print(X_test)
#print(y_train)
#print(y_test)

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import ExtraTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.preprocessing import StandardScaler

# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)


models = {
    'Logistic Regression': LogisticRegression(max_iter=5000,solver = 'saga'),  # Increased max_iter
    'Naive Bayes': GaussianNB(),
    'Support Vector Machine': SVC(),  # Consider setting probability=True if needed
    'K-Nearest Neighbors': KNeighborsClassifier(),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'Bagging': BaggingClassifier(),
    'AdaBoost': AdaBoostClassifier(algorithm="SAMME"),  # Set SAMME explicitly
    'Gradient Boosting': GradientBoostingClassifier(),
    'Extra Trees': ExtraTreeClassifier(),
}

# for name,model in models.items():
#     model.fit(X_train_scaled ,y_train)
#     ypred= model.predict(X_test_scaled)

#print(f"{name}  with accuracy : {accuracy_score(y_test,ypred)}")
#print("========================================================")

rfc = RandomForestClassifier()
rfc.fit(X_train,y_train)
ypred= rfc.predict(X_test)
accuracy_score(y_test,ypred)

import pickle

pickle.dump(rfc,open('croprecmodel.pkl','wb'))


