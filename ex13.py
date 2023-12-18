from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,classification_report
data=load_breast_cancer()
x=data.data#features
y=data.target#targets
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
dt=DecisionTreeClassifier()
dt.fit(x_train,y_train)
print(dt.predict(x_test))
V=dt.predict(x_test)
result=accuracy_score(y_test,V)
report=classification_report(y_test,V)
print("accuracy:",result)
print("\nclassification report:\n",report)
