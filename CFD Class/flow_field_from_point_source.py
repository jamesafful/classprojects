import numpy as np
import matplotlib.pyplot as plt

# Parameters for the flow
U = 1  # Free stream speed
Q = 1  # Source strength

# Define the range for R and Z
R_range = np.linspace(-2, 2, 100)
Z_range = np.linspace(-2, 2, 100)

# Create meshgrid
R, Z = np.meshgrid(R_range, Z_range)

# Potential function phi
with np.errstate(divide='ignore', invalid='ignore'):
    phi = U * Z - (Q / (4 * np.pi * np.sqrt(Z**2 + R**2)))

# Velocity components in R and Z directions
# Derivative of potential function with respect to R and Z gives velocity components
V_R = np.gradient(phi, R_range, axis=1)
V_Z = np.gradient(phi, Z_range, axis=0)

# Plot streamlines
plt.figure(figsize=(8, 6))
strm = plt.streamplot(R, Z, V_R, V_Z, color='black', linewidth=1, arrowsize=1.5)

plt.xlabel('R')
plt.ylabel('Z')
plt.title('Streamlines in R-Z Half Plane for Point Source in Free Stream')
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.xlim([R_range[0], R_range[-1]])
plt.ylim([Z_range[0], Z_range[-1]])
plt.grid(True)
plt.show()
