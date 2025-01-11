import numpy as np
import matplotlib.pyplot as plt

def ADI(U, dt, dx, T):
    """
    Solves the 2D heat equation u,t = u,xx + u,yy using the ADI scheme
    
    Parameters:
    U (ndarray): initial condition
    dt (float): time step
    dx (float): spatial step
    T (float): final time
    
    Returns:
    U (ndarray): solution
    """
    
    # Define constants
    m, n = U.shape
    alpha = dt/dx**2
    
    # Construct matrices
    Mx = np.zeros((n-2, n-2))
    My = np.zeros((m-2, m-2))
    for j in range(1, n-2):
        Mx[j-1,j-1] = 1 + 2*alpha
        Mx[j-1,j] = -alpha
        Mx[j,j-1] = -alpha
    for i in range(1, m-2):
        My[i-1,i-1] = 1 + 2*alpha
        My[i-1,i] = -alpha
        My[i,i-1] = -alpha
    
    # ADI scheme
    t = 0
    while t < T:
        # x-sweep
        for i in range(1, m-2):
            B = U[i,1:n-1]
            B[0] += alpha*U[i,0]
            B[n-3] += alpha*U[i,n-1]
            U[i,1:n-1] = np.linalg.solve(Mx, B)
        # y-sweep
        for j in range(1, n-2):
            B = U[1:m-1,j]
            B[0] += alpha*U[0,j]
            B[m-3] += alpha*U[m-1,j]
            U[1:m-1,j] = np.linalg.solve(My, B)
        t += dt
    
    return U


# Define parameters
N = 100
T = 0.2
dt = 0.001
dx = 1/(N-1)

# Create grid
x = np.linspace(0, 1, N)
y = np.linspace(0, 1, N)
X, Y = np.meshgrid(x, y)

# Define initial condition
f = np.sin(np.pi*X)*np.sin(np.pi*Y)
U = f.copy()

# Set boundary condition
U[:,0] = 0
U[:,-1] = 0
U[0,:] = 0
U[-1,:] = 0

# Compute solution
U = ADI(U, dt, dx, T)

# Plot solution
#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#ax.plot_surface(X, Y, U, cmap='viridis')
#ax.set_xlabel('x')
#ax.set_ylabel('y')
#ax.set_zlabel('u')
#plt.show()



#Plot
plt.pcolormesh(x, y, U)
plt.colorbar()
plt.title('Solution for 2-D Heat equation at time T=0.01')
plt.xlabel('x-axis');
plt.ylabel('Y-axis');
plt.show()

