


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets





class percepton():
    def __init__(self):
        self.w=None
        self.b=None
    def model(self,x):
        return 1 if (np.dot(self.w,x)>=self.b) else 0
    def predict(self ,X):
        Y=[]
        for x in X:
            result =self.model(x)
            Y.append(result)
        return np.array(Y)  
    def fit(self,X,Y,epochs=1):
        self.w=np.ones(X.shape[1])
        self.b=0
        for i in range(epochs):
            for x,y in zip(X,Y):
                y_pred=self.model(x)
                if (y==1)and (y_pred==0):
                    self.w=self.w+x
                    self.b=self.b+1
                elif (y==0) and (y_pred==1):
                    self.w=self.w-x
                    self.b=self.b-1
            

                
            


neuron=percepton()
bstr=datasets.load_breast_cancer()
from sklearn.model_selection import train_test_split 
data=pd.DataFrame(bstr.data,columns=bstr.feature_names)
data["class"]=bstr.target
X=data.drop("class",axis=1)
Y=data["class"]
X_train,X_test,Y_train,Y_test=train_test_split(X,Y, test_size=0.1,stratify=Y,random_state=1)
X_train=X_train.values
X_test=X_test.values
neuron.fit(X_train,Y_train,100)
plt.plot(neuron.w)
plt.show()
from sklearn.metrics  import  accuracy_score
y_pred=neuron.predict(X_train)
accuracy=accuracy_score(y_pred,Y_train)







