from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd 
import pickle

#load the dataset
iris = load_iris()
X = iris.data
y = iris.target

#print(X[-2:],y[-2:])

seed = 42
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=seed,test_size = 0.33)


#build the model
clf = LogisticRegression()

#train
clf.fit(X_train,y_train)

#predict
predicted = clf.predict(X_test)

#accuracy
print(accuracy_score(predicted,y_test))

# save the model to disk
filename = 'rf_model.pkl'
pickle.dump(clf, open(filename, 'wb'))




