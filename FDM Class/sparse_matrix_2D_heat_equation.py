import numpy as np
import matplotlib;
import matplotlib.pylab as plt;
import scipy.sparse as sp;
from scipy.sparse import spdiags;
from scipy.sparse.linalg import spsolve;
from scipy.sparse import csc_matrix;
from numpy import linalg;

N = 19

# Create an N x N mesh over the extended domain [0,1]^2
x = np.linspace(0, 1, N)
y = np.linspace(0, 1, N)

# Initialize the coefficient matrix A
A = np.zeros((N**2, N**2))

# Generate the coefficient matrix A
for i in range(N**2):
    row = i // N
    col = i % N
    
    if (x[col] >= 0 and x[col] <= 0.5 and y[row] >= 0.5 and y[row] <= 1) or (y[row] == 0.5 and 0 <= x[col] <= 1) or (x[col] == 0.5 and 0 <= y[row] <= 1):
        A[i, i] = 1
    else:
        A[i, i] = 4
        if row > 0:
            A[i, i - N] = -1
        if row < N - 1:
            A[i, i + N] = -1
        if col > 0:
            A[i, i - 1] = -1
        if col < N - 1:
            A[i, i + 1] = -1

A = csc_matrix(A)
A = A.todense()
print(A)

# Plot the sparsity patterns
#fig, (ax1, ax2) = plt.subplots(1, 2)
#ax1.spy(A, markersize=3)
#ax1.set_title('N = 19')


plt.title(r'Sparse matrix pattern 2-D Heat Equation N=19')
plt.xlabel(r'Column')
plt.ylabel(r'Row')
plt.spy(A, markersize=2, color='red')
plt.show()