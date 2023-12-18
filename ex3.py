import numpy as np
matrix1=np.array([[1,2],[3,4]])
matrix2=np.array([[5,6],[7,8]])

print(matrix1)
print(matrix2)
sum=np.add(matrix1,matrix2)

pdt=np.multiply(matrix1,matrix2)

quotient=np.divide(matrix1,matrix2)

dtp=np.dot(matrix1,matrix2)

trans1=np.transpose(matrix1)

trans2=np.transpose(matrix2)

diff=np.subtract(matrix1,matrix2)

print("sum =",sum)
print("pdt =",pdt)
print("quotient =",quotient)
print("dot product =",dtp)
print("transpose =",trans1)
print("transpose =",trans2)
print("difference=",diff)