#Gender classification using universal data set

#import libaries
import numpy as np
import pandas as pd
#ML Packages
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer
#load teh data set
data_set = pd.read_csv("names_dataset.csv")
xfeatures  = data_set["name"]

#feature extraction
cv = CountVectorizer()
X = cv.fit_transform(xfeatures)
print(cv.get_feature_names())

data_set.sex.replace({'F':0,'M':1},inplace=True)

#features
X = X
#label
data_set.drop_duplicates(keep="first", inplace=True)
y  =data_set.sex
from collections import  Counter
print("ty",Counter(y))
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# Naive Bayes Classifier
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB()
clf.fit(X_train,y_train)
clf.score(X_test,y_test)


# Accuracy of our Model
print("Accuracy of Model",clf.score(X_test,y_test)*100,"%")

# Sample1 Prediction
sample_name = ["Sohan"]
vect = cv.transform(sample_name).toarray()
# Female is 0, Male is 1

# A function to do it
def genderpredictor(a):
    test_name = [a]
    vector = cv.transform(test_name).toarray()
    print(clf.predict(vector))

    if clf.predict(vector) == 0:
        print("Female")
    else:
        print("Male")

genderpredictor("richa")

#Alternative to Model Saving
import pickle
dctreeModel = open("namesdetectormodel.pkl","wb")

pickle.dump(clf,dctreeModel)
