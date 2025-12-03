import gmsh
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time 
#CREATION OF A 2D BEAM MESH USING GMSH
gmsh.initialize()
gmsh.model.add("beam_only")

# Mesh size
lc = 0.5 #[mm]

# Beam geometry (you can change these)
beam_width  = 5     # width in x [mm]
beam_height = 5    # height in y [mm]

xL = beam_width- beam_width/2   # left x
xR = beam_width + beam_width/2   # right x
yB = 0.0                  # bottom
yT = beam_height          # top

# Points (counter-clockwise)
p1 = gmsh.model.geo.addPoint(xL, yB, 0, lc, 1)
p2 = gmsh.model.geo.addPoint(xR, yB, 0, lc, 2)
p3 = gmsh.model.geo.addPoint(xR, yT, 0, lc, 3)
p4 = gmsh.model.geo.addPoint(xL, yT, 0, lc, 4)

# Lines
l1 = gmsh.model.geo.addLine(1, 2)
l2 = gmsh.model.geo.addLine(2, 3)
l3 = gmsh.model.geo.addLine(3, 4)
l4 = gmsh.model.geo.addLine(4, 1)

# Surface
loop_beam = gmsh.model.geo.addCurveLoop([l1, l2, l3, l4])
surf_beam = gmsh.model.geo.addPlaneSurface([loop_beam], 1)

gmsh.model.geo.synchronize()

# Physical group for the beam
gmsh.model.addPhysicalGroup(2, [surf_beam], 1)
gmsh.model.setPhysicalName(2, 1, "Beam")

# Mesh
gmsh.option.setNumber("Mesh.ElementOrder", 1)
gmsh.option.setNumber("Mesh.Algorithm", 6)
gmsh.model.mesh.generate(2)

gmsh.write("beam_only.msh")
gmsh.finalize()


# VISUALIZATION OF THE MESH USING MATPLOTLIB
gmsh.initialize()
gmsh.open("beam_only.msh")   # <-- new mesh file

# Nodes
node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
node_coords = np.array(node_coords).reshape(-1, 3)

def get_elements_of_physical_group(dim, tag):
    entity_tags = gmsh.model.getEntitiesForPhysicalGroup(dim, tag)
    all_elements = []
    for entity_tag in entity_tags:
        elem_types, elem_tags, elem_node_tags = gmsh.model.mesh.getElements(dim, entity_tag)
        for i in range(len(elem_types)):
            all_elements.append(np.array(elem_node_tags[i]).reshape(-1, 3))
    return np.concatenate(all_elements) if all_elements else np.array([])

elements_beam = get_elements_of_physical_group(2, 1)
'''
plt.figure(figsize=(6, 8))
if elements_beam.size > 0:
    plt.triplot(
        node_coords[:, 0],
        node_coords[:, 1],
        elements_beam - 1,
        color='blue',
        label='Beam'
    )

plt.gca().set_aspect('equal')
plt.title('Beam mesh')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()

gmsh.finalize()
'''

coord = node_coords[:,:2]



def jacobian(coord):
 
  return np.abs((coord[0,0]-coord[0,2]) * (coord[1,1]-coord[1,2]) - (coord[0,1] - coord[0,2]) * (coord[1,0] - coord[1,2]))

def B_matrix(coord):

    x1, x2, x3 = coord[0]
    y1, y2, y3 = coord[1]
    Α2 = x1*(y2 - y3) + x2*(y3 - y1) + x3*(y1 - y2)
    # Derivatives of shape functions
    dNdx = np.array([(y2 - y3) / Α2,(y3 - y1) / Α2,(y1 - y2) / Α2 ])

    dNdy = np.array([  (x3 - x2) / Α2, (x1 - x3) / Α2, (x2 - x1) / Α2])
    B = np.zeros((3, 6))
    B[0, 0::2] = dNdx          # ε_xx
    B[1, 1::2] = dNdy          # ε_yy
    B[2, 0::2] = dNdy          # γ_xy (u_x part)
    B[2, 1::2] = dNdx          # γ_xy (u_y part)

    return B


def D_matrix(E, nu, plane_stress=False):  #σ=Dε
    if plane_stress:
        coef = E / (1.0 - nu**2)
        D = coef * np.array([
            [1.0,   nu,           0.0],
            [nu,    1.0,          0.0],
            [0.0,   0.0, (1.0 - nu)/2.0]
        ])
    else:
        coef = E / ((1.0 + nu)*(1.0 - 2.0*nu))
        D = coef * np.array([
            [1.0 - nu,    nu,           0.0],
            [nu,          1.0 - nu,     0.0],
            [0.0,         0.0, 0.5*(1.0 - 2.0*nu)]
        ])
    return D


def CST_elastic(coord, E, nu, t=1.0, plane_stress=False):

    A = jacobian(coord) / 2.0
    B = B_matrix(coord)  
    D = D_matrix(E, nu, plane_stress)

    Ke = t * A * (B.T @ D @ B)
    return Ke


 #Consistent elastoplastic tangent for isotropic von Mises plasticity
def compute_Cep(C, sigma_trial, sigma_y,alpha,  H):
    s = sigma_trial - np.mean(sigma_trial[:2])*np.array([1,1,0])
    norm_s = np.linalg.norm(s) * np.sqrt(1.5)
    f_trial = norm_s - (sigma_y + H*alpha)
    if f_trial <= 0:
        #return C, _  # elastic
        return C # elastic
    else:
        #print('Ayayaya')
        n = s / np.linalg.norm(s)
        #print(n)
        n = n.reshape(3,1)
        Ce = C - (C @ n @ n.T @ C) / (n.T @ C @ n + H)
        #return Ce, n
        return Ce
    
# --- Helper functions for plasticity calculations (assuming plane strain) ---
# Definition of a von Mises stress calculation for a stress vector [sigma_x, sigma_y, tau_xy]

def calculate_von_mises_stress(stress_vector, nu):
    sigma_x, sigma_y, tau_xy = stress_vector
    sigma_z = nu * (sigma_x + sigma_y) # Plane strain
    return np.sqrt(
        sigma_x**2 + sigma_y**2 + sigma_z**2
        - sigma_x * sigma_y - sigma_y * sigma_z - sigma_z * sigma_x
        + 3 * tau_xy**2 )

# Definition of deviatoric stress components for a stress vector [sigma_x, sigma_y, tau_xy]
def deviatoric_part(stress_vector, nu):
    sigma_x, sigma_y, tau_xy = stress_vector
    sigma_zz = nu * (sigma_x + sigma_y) # Plane strain
    p = (sigma_x + sigma_y + sigma_zz) / 3.0
    Sx = sigma_x - p
    Sy = sigma_y - p
    # Sz = sigma_zz - p # Not explicitly needed for 2D Voigt vector output
    return np.array([Sx, Sy, tau_xy])

# Definition of hydrostatic pressure for a stress vector [sigma_x, sigma_y, tau_xy]
def hydrostatic_part(stress_vector, nu):
    sigma_x, sigma_y, tau_xy = stress_vector
    sigma_zz = nu * (sigma_x + sigma_y) # Plane strain
    return (sigma_x + sigma_y + sigma_zz) / 3.0


# Yield function (adapted from cH4cZuiTJO1A, now takes required parameters as arguments)
def yield_function_plasticity(stress_vector, E_val, nu_val, sigma_y0_val, kappa_val, H_val):
    sigma_x = stress_vector[0]
    sigma_y = stress_vector[1]
    tau_xy  = stress_vector[2]
    sigma_zz = nu_val * (sigma_x + sigma_y)  # plane strain

    sigma_tensor = np.array([
        [sigma_x, tau_xy, 0.0],
        [tau_xy, sigma_y, 0.0],
        [0.0,    0.0,    sigma_zz] ])

    p = np.trace(sigma_tensor) / 3.0
    I_3d = np.identity(3)
    S_tr_matrix = sigma_tensor - p * I_3d

    S_dot_S = np.tensordot(S_tr_matrix, S_tr_matrix, axes=2)
    sigma_eq_tr = np.sqrt(1.5 * S_dot_S)

    sigma_y_curr = sigma_y0_val + H_val * kappa_val
    f_tr = sigma_eq_tr - sigma_y_curr
    return f_tr


num_nodes = len(node_coords)       # <– not len(2*node_coords)
n_elem    = len(elements_beam)

# Material parameters
E  = 30e3
nu = 0.3
G  = E / (2.0 * (1.0 + nu))
sigma_y0 = 50
Et =- E / 100.0
H  = E * Et / (E - Et)

# Internal variables
kappa = np.zeros(n_elem)
sigma_eq = np.zeros(n_elem)
yield_func_history = np.zeros(n_elem)
current_stress_per_element = np.zeros((n_elem, 3))

### displacement control plasticity

num_dofs  = 2 * num_nodes
num_steps = 10
U = np.zeros(num_dofs)
max_iter = 30

u_max = -0.01  # imposed top displacement (downwards)

# --- identify top and bottom nodes ONCE ---
y_coords = node_coords[:, 1]
y_max = y_coords.max()
y_min = y_coords.min()
tol = 1e-8

top_nodes    = np.where(np.isclose(y_coords, y_max, atol=tol))[0]
bottom_nodes = np.where(np.isclose(y_coords, y_min, atol=tol))[0]

# DOFs with imposed displacement (top, vertical)
disp_dofs = 2 * top_nodes + 1        # v-DOFs on top

# DOFs of clamped end at bottom (both x and y)
bottom_x_dofs = 2 * bottom_nodes
bottom_y_dofs = 2 * bottom_nodes + 1
bottom_dofs   = np.concatenate((bottom_x_dofs, bottom_y_dofs))

fixed_dofs = bottom_dofs            # clamped end

# All constrained DOFs = clamped bottom + imposed displacement on top
constrained_dofs = np.union1d(fixed_dofs, disp_dofs)
free_dofs        = np.setdiff1d(np.arange(num_dofs), constrained_dofs)

# arrays for F–u curve (using top displacement vs bottom reaction)
sum_R  = np.zeros(num_steps)  # resultant vertical reaction at bottom
sum_du = np.zeros(num_steps)  # imposed displacement at top


for step in range(num_steps):
    print(f'Load step ({step+1})/{num_steps}')

    # imposed displacement at this step
    u_step = u_max * (step + 1) / num_steps

    # initial guess with BCs
    U[fixed_dofs] = 0.0
    U[disp_dofs]  = u_step

    iter = 0
    R = None   # will hold residual for reactions at convergence

    while iter < max_iter:
        K = np.zeros((num_dofs, num_dofs))
        F = np.zeros(num_dofs)   # no externally applied forces
        R = F.copy()             # R = F - f_int = -f_int

        for e in range(n_elem):
            n1, n2, n3 = elements_beam[e] - 1
            element_coord = np.array([
                [node_coords[n1, 0], node_coords[n2, 0], node_coords[n3, 0]],
                [node_coords[n1, 1], node_coords[n2, 1], node_coords[n3, 1]],
            ])
            B_e = B_matrix(element_coord)
            A   = jacobian(element_coord) / 2.0

            edof = np.array([
                2*n1, 2*n1 + 1,
                2*n2, 2*n2 + 1,
                2*n3, 2*n3 + 1
            ])
            u_e = U[edof]
            eps_e = B_e @ u_e

            # --- elastic predictor + return mapping ---
            sigma_trial_vector = D_matrix(E, nu, plane_stress=False) @ eps_e
            f_tr = yield_function_plasticity(
                sigma_trial_vector, E, nu, sigma_y0, kappa[e], H
            )

            if f_tr <= 0.0:
                sigma_new_vector = sigma_trial_vector
                kappa_new = kappa[e]
                Cep = D_matrix(E, nu, plane_stress=False)
            else:
                
                S_tr_vector = deviatoric_part(sigma_trial_vector, nu)
                sigma_eq_tr = calculate_von_mises_stress(sigma_trial_vector, nu)
                n_vec = 1.5 * S_tr_vector / sigma_eq_tr

                dgamma = f_tr / (3.0 * G + H)
                
                sigma_new_vector = sigma_trial_vector - 2.0 * G * dgamma * n_vec
                kappa_new = kappa[e] +np.sqrt(2/3) *dgamma

                Cel = D_matrix(E, nu, plane_stress=False)
                #Cep = compute_Cep(Cel, sigma_new_vector, sigma_y0,kappa_new, H)
               # Cep=compute_Cep_new(Cel, sigma_new_vector, H)
                Cep=Cel
                
            kappa[e] = kappa_new
            current_stress_per_element[e] = sigma_new_vector
            sigma_eq[e] = calculate_von_mises_stress(sigma_new_vector, nu)
            yield_func_history[e] = sigma_eq[e] - (sigma_y0 + H * kappa[e])

            Ke      = B_e.T @ Cep @ B_e * A
            f_int_e = B_e.T @ sigma_new_vector * A

            for i_local in range(6):
                I_global = edof[i_local]
                R[I_global] -= f_int_e[i_local]    # F - f_int
                for j_local in range(6):
                    J_global = edof[j_local]
                    K[I_global, J_global] += Ke[i_local, j_local]

        # --- solve only for free DOFs ---
        K_ff = K[np.ix_(free_dofs, free_dofs)]
        R_f  = R[free_dofs]

        delta_U = np.zeros_like(U)
        delta_U[free_dofs] = np.linalg.solve(K_ff, R_f)

        U[free_dofs] += delta_U[free_dofs]

        # re-enforce BCs
        U[fixed_dofs] = 0.0
        U[disp_dofs]  = u_step

        res_norm = np.linalg.norm(delta_U[free_dofs])
        print(res_norm)
        if res_norm < 1e-5:
            print(f'  Newton converged in {iter+1} iterations')
            break

        iter += 1

    # ===== reactions at clamped bottom after convergence =====
    # R = F - f_int, and F=0 -> R = -f_int.
    # So reaction forces at supports (bottom) are:
    reactions_bottom = R[bottom_dofs]

    # If you want the RESULTANT vertical reaction at the clamped end:
    reactions_bottom_y = R[bottom_y_dofs]
    # sign: R is "external - internal" → we often take minus to plot "applied force"
    resultant_vertical_reaction = -np.sum(reactions_bottom_y)

    sum_R[step]  = resultant_vertical_reaction
    sum_du[step] = u_step