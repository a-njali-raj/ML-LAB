import numpy as np
#your matrix
matrix =np.array([[5,6,4],
                  [2,5,6],
                  [3,5,6]])
#perform SVD
U,S,VT= np.linalg.svd(matrix)
print("U matrix:")
print(U)

print("S matrix(Singular values):")
print(np.diag(S))

print("VT matrix:")
print(VT)

reconstructed_matrix=np.dot(U,np.dot(np.diag(S),VT))
print("reconstructed matrix")
print(reconstructed_matrix)
