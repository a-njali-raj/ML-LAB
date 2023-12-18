from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split#train-test split function
from sklearn.datasets import load_digits# needed dataset imported
from sklearn.metrics import accuracy_score#measurement of accuracy while comparing model with test data
digits=load_digits()
x=digits.data#features
y=digits.target#targets
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)#split testand train correspondent with features and targets in 80 and 20%
knn=KNeighborsClassifier(n_neighbors=7)#k=7 for  model prediction
knn.fit(x_train,y_train)
print(knn.predict(x_test))
V=knn.predict(x_test)#model is assigned to V
result=accuracy_score(y_test,V)
print("accuracy:",result)
