import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg
import matplotlib.pyplot as plt
import pandas as pd

def generate_mesh(Nx, Ny, xmin=-1, xmax=1, ymin=-1, ymax=1):
    x = np.linspace(xmin, xmax, Nx+1)
    y = np.linspace(ymin, ymax, Ny+1)
    xv, yv = np.meshgrid(x, y, indexing='ij')
    nodes = np.column_stack([xv.ravel(), yv.ravel()])
    elements = []
    for i in range(Nx):
        for j in range(Ny):
            n1 = i*(Ny+1) + j
            n2 = (i+1)*(Ny+1) + j
            n3 = (i+1)*(Ny+1) + (j+1)
            n4 = i*(Ny+1) + (j+1)
            elements.append([n1, n2, n3, n4])  # Node ordering: bottom-left, bottom-right, top-right, top-left
    return nodes, np.array(elements)

def shape_functions(xi, eta):
    N = np.array([
        (1 - xi)*(1 - eta)/4,  # N1
        (1 + xi)*(1 - eta)/4,  # N2
        (1 + xi)*(1 + eta)/4,  # N3
        (1 - xi)*(1 + eta)/4   # N4
    ])
    dN_dxi = np.array([
        [-(1 - eta)/4, -(1 - xi)/4],  # dN1/dxi, dN1/deta
        [ (1 - eta)/4, -(1 + xi)/4],  # dN2/dxi, dN2/deta
        [ (1 + eta)/4,  (1 + xi)/4],  # dN3/dxi, dN3/deta
        [-(1 + eta)/4,  (1 - xi)/4]   # dN4/dxi, dN4/deta
    ])
    return N, dN_dxi

def gauss_quadrature_edge():
    # 2-point Gauss quadrature for edges
    points = np.array([-1/np.sqrt(3), 1/np.sqrt(3)])
    weights = np.array([1, 1])
    return points, weights

def gauss_quadrature():
    # 2-point Gauss quadrature
    points = np.array([-1/np.sqrt(3), 1/np.sqrt(3)])
    weights = np.array([1, 1])
    return points, weights

def assemble_system(nodes, elements, f_func, g_func):
    num_nodes = nodes.shape[0]
    K = sp.lil_matrix((num_nodes, num_nodes))
    F = np.zeros(num_nodes)
    # Gauss quadrature points and weights
    gauss_pts, gauss_wts = gauss_quadrature()
    gauss_pts_edge, gauss_wts_edge = gauss_quadrature_edge()
    for elem in elements:
        Ke = np.zeros((4, 4))
        Fe = np.zeros(4)
        # Node indices and coordinates
        node_indices = elem
        coords = nodes[node_indices]
        # Element stiffness matrix and load vector
        for i in range(len(gauss_pts)):
            xi = gauss_pts[i]
            wi = gauss_wts[i]
            for j in range(len(gauss_pts)):
                eta = gauss_pts[j]
                wj = gauss_wts[j]
                N, dN_dxi = shape_functions(xi, eta)
                J = dN_dxi.T @ coords
                detJ = np.linalg.det(J)
                if detJ <= 0:
                    raise ValueError("Negative or zero determinant of Jacobian encountered.")
                invJ = np.linalg.inv(J)
                dN_dx = dN_dxi @ invJ
                x_y = N @ coords
                f_val = f_func(x_y[0], x_y[1])
                Ke += (dN_dx @ dN_dx.T) * detJ * wi * wj
                Fe += N * f_val * detJ * wi * wj
        # Assemble into global matrix and vector
        for a in range(4):
            A = node_indices[a]
            F[A] += Fe[a]
            for b in range(4):
                B = node_indices[b]
                K[A, B] += Ke[a, b]
        # Boundary integration for Robin BCs
        edge_nodes = [
            (0, 1, -1, 'eta'),  # Bottom edge at eta = -1
            (1, 2, 1, 'xi'),    # Right edge at xi = +1
            (2, 3, 1, 'eta'),   # Top edge at eta = +1
            (3, 0, -1, 'xi'),   # Left edge at xi = -1
        ]
        for edge in edge_nodes:
            n1_local, n2_local, const, var = edge
            n1_global = node_indices[n1_local]
            n2_global = node_indices[n2_local]
            x1, y1 = nodes[n1_global]
            x2, y2 = nodes[n2_global]
            # Check if the edge is on the boundary
            if (abs(x1 - (-1)) < 1e-10 and abs(x2 - (-1)) < 1e-10) or \
               (abs(x1 - 1) < 1e-10 and abs(x2 - 1) < 1e-10) or \
               (abs(y1 - (-1)) < 1e-10 and abs(y2 - (-1)) < 1e-10) or \
               (abs(y1 - 1) < 1e-10 and abs(y2 - 1) < 1e-10):
                # Compute edge length
                edge_length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                J_edge = edge_length / 2  # Jacobian determinant for edge mapping
                # Perform boundary integration
                for i in range(len(gauss_pts_edge)):
                    s = gauss_pts_edge[i]
                    ws = gauss_wts_edge[i]
                    if var == 'xi':
                        xi = const
                        eta = s
                    else:
                        xi = s
                        eta = const
                    N_edge, _ = shape_functions(xi, eta)
                    x_y = N_edge @ coords
                    g_val = g_func(x_y[0], x_y[1])
                    weight = ws * J_edge
                    # Update K and F
                    for a in range(4):
                        A = node_indices[a]
                        F[A] += N_edge[a] * g_val * weight
                        for b in range(4):
                            B = node_indices[b]
                            K[A, B] += N_edge[a] * N_edge[b] * weight
    return K.tocsr(), F

def f_function(x, y):
    pi = np.pi
    u = np.exp(np.sin(pi*x)*np.sin(pi*y))
    sin_pi_x = np.sin(pi*x)
    sin_pi_y = np.sin(pi*y)
    cos_pi_x = np.cos(pi*x)
    cos_pi_y = np.cos(pi*y)
    term1 = 2*sin_pi_x*sin_pi_y
    term2 = 2*cos_pi_x**2*cos_pi_y**2
    term3 = -cos_pi_x**2 - cos_pi_y**2
    return pi**2 * u * (term1 + term2 + term3)

def g_function(x, y):
    pi = np.pi
    tol = 1e-10
    if abs(x - (-1)) < tol:
        return 1 + pi * np.sin(pi*y)
    elif abs(x - 1) < tol:
        return 1 - pi * np.sin(pi*y)
    elif abs(y - (-1)) < tol:
        return 1 + pi * np.sin(pi*x)
    elif abs(y - 1) < tol:
        return 1 - pi * np.sin(pi*x)
    else:
        return 0.0

def exact_solution(x, y):
    return np.exp(np.sin(np.pi*x)*np.sin(np.pi*y))

def grad_exact_solution(x, y):
    pi = np.pi
    u = exact_solution(x, y)
    sin_pi_x = np.sin(pi*x)
    cos_pi_x = np.cos(pi*x)
    sin_pi_y = np.sin(pi*y)
    cos_pi_y = np.cos(pi*y)
    du_dx = u * pi * cos_pi_x * sin_pi_y
    du_dy = u * pi * sin_pi_x * cos_pi_y
    return np.array([du_dx, du_dy])

def compute_errors(nodes, elements, U, exact_solution, grad_exact_solution):
    L2_error = 0.0
    H1_error = 0.0
    gauss_pts, gauss_wts = gauss_quadrature()
    for elem in elements:
        node_indices = elem
        coords = nodes[node_indices]
        U_e = U[node_indices]
        for i in range(len(gauss_pts)):
            xi = gauss_pts[i]
            wi = gauss_wts[i]
            for j in range(len(gauss_pts)):
                eta = gauss_pts[j]
                wj = gauss_wts[j]
                N, dN_dxi = shape_functions(xi, eta)
                J = dN_dxi.T @ coords
                detJ = np.linalg.det(J)
                if detJ <= 0:
                    raise ValueError("Negative or zero determinant of Jacobian encountered.")
                invJ = np.linalg.inv(J)
                dN_dx = dN_dxi @ invJ
                x_y = N @ coords
                u_exact = exact_solution(x_y[0], x_y[1])
                grad_u_exact = grad_exact_solution(x_y[0], x_y[1])
                u_h = N @ U_e
                grad_u_h = dN_dx.T @ U_e
                L2_error += ((u_exact - u_h)**2) * detJ * wi * wj
                H1_error += (np.linalg.norm(grad_u_exact - grad_u_h)**2) * detJ * wi * wj
    L2_error = np.sqrt(L2_error)
    H1_error = np.sqrt(H1_error)
    return L2_error, H1_error

def main():
    hs = [1/4, 1/8, 1/16, 1/32]
    L2_errors = []
    H1_errors = []
    for h in hs:
        Nx = Ny = int(2 / h)
        nodes, elements = generate_mesh(Nx, Ny)
        K, F = assemble_system(nodes, elements, f_function, g_function)
        # Solve the linear system
        U = scipy.sparse.linalg.spsolve(K, F)
        # Compute errors
        L2_err, H1_err = compute_errors(nodes, elements, U, exact_solution, grad_exact_solution)
        L2_errors.append(L2_err)
        H1_errors.append(H1_err)
        print(f'h = {h:.5f}, L2 error = {L2_err:.5e}, H1 error = {H1_err:.5e}')
    # Compute convergence rates
    def convergence_rates(errors, hs):
        rates = []
        for i in range(1, len(errors)):
            rate = np.log(errors[i]/errors[i-1]) / np.log(hs[i]/hs[i-1])
            rates.append(rate)
        return rates
    L2_rates = convergence_rates(L2_errors, hs)
    H1_rates = convergence_rates(H1_errors, hs)
    # Create convergence table using pandas
    data = {
        'h': hs,
        'L2 Error': L2_errors,
        'L2 Order': ['-'] + [f"{r:.2f}" for r in L2_rates],
        'H1 Error': H1_errors,
        'H1 Order': ['-'] + [f"{r:.2f}" for r in H1_rates],
    }
    df = pd.DataFrame(data)
    print('\nConvergence Table:')
    print(df)
    # Save convergence table as an image
    try:
        import dataframe_image as dfi  # type: ignore
        df_styled = df.style.set_caption("Convergence Table").format(precision=5)
        #dfi.export(df_styled, 'convergence_table.png')
        print("\nConvergence table saved as 'convergence_table.png'.")
    except ImportError:
        print("\nTo save the convergence table as an image, please install 'dataframe_image' via 'pip install dataframe_image'.")
    # Plotting the convergence
    plt.figure(figsize=(8, 6))
    plt.loglog(hs, L2_errors, 'o-', label='$L^2$ Error')
    plt.loglog(hs, H1_errors, 's-', label='$H^1$ Error')
    # Reference lines
    C_L2 = L2_errors[0] / hs[0]**2
    C_H1 = H1_errors[0] / hs[0]**1
    hs_fine = np.linspace(hs[-1], hs[0], 100)
    plt.loglog(hs_fine, C_L2 * hs_fine**2, 'k--', label='$O(h^2)$ Reference')
    plt.loglog(hs_fine, C_H1 * hs_fine, 'k-.', label='$O(h)$ Reference')
    plt.gca().invert_xaxis()
    plt.xlabel('Mesh size $h$')
    plt.ylabel('Error Norm')
    plt.title('Convergence Plot')
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.savefig('convergence_plot.png', dpi=300)
    print("Convergence plot saved as 'convergence_plot.png'.")
    plt.show()
    # Plot the approximate solution
    plt.figure()
    plt.tripcolor(nodes[:,0], nodes[:,1], U, shading='gouraud')
    plt.colorbar()
    plt.title('Approximate Solution U')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig('approximate_solution.png', dpi=300)
    print("Approximate solution plot saved as 'approximate_solution.png'.")
    plt.show()
    # Plot the error
    num_nodes = nodes.shape[0]
    U_exact = np.array([exact_solution(nodes[i,0], nodes[i,1]) for i in range(num_nodes)])
    error = U_exact - U
    plt.figure()
    plt.tripcolor(nodes[:,0], nodes[:,1], error, shading='gouraud')
    plt.colorbar()
    plt.title('Error U_exact - U')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig('error_plot.png', dpi=300)
    print("Error plot saved as 'error_plot.png'.")
    plt.show()

if __name__ == "__main__":
    main()
