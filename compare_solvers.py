import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
import warnings
import time
from scipy.optimize import minimize

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# --------------------------------------------------------------------------------
# PART 1: SHARED SYSTEM PARAMETERS
# --------------------------------------------------------------------------------

# Global Constants
m0 = 2000.0              # initial mass (kg)
mf = 300.0               # fuel mass (kg)
alpha = 5e-4             # fuel consumption rate (s/m)
Tmax = 24000.0           # max thrust (N)
rho1 = 0.2 * Tmax        # lower bound thrust (N)
rho2 = 0.8 * Tmax        # upper bound thrust (N)
velocity_max = 90.0      # max velocity (m/s)
glidelslope_angle = 30.0 # degrees

# Planet Parameters (Mars)
omega = np.array([2.53e-5, 0, 6.62e-5]) 
g = np.array([-3.71, 0, 0])             

# Unit vectors
e1 = np.array([1, 0, 0]) 
e2 = np.array([0, 1, 0])
e3 = np.array([0, 0, 1])

# Matrices for Dynamics
S = np.array([[0, -omega[2], omega[1]], 
              [omega[2], 0, -omega[0]], 
              [-omega[1], omega[0], 0]])

# A_cont and B_cont
A_cont = np.block([[np.zeros((3, 3)), np.eye(3)], 
                   [-S @ S, -2 * S]])
B_cont = np.block([[np.zeros((3, 3))], 
                   [np.eye(3)]])

# E matrix for position error (Y-Z plane error)
E_mat = np.array([[0, 1, 0, 0, 0, 0], 
                  [0, 0, 1, 0, 0, 0]])

# --------------------------------------------------------------------------------
# PART 2: CVXPY SOLVER LOGIC
# --------------------------------------------------------------------------------

def cvx_get_constraints(x, z, u, gamma, N, dt, x_init, theta_cos, glide_tan):
    """Constructs the list of convex constraints for CVXPY."""
    constraints = []
    
    # 1. Boundary Conditions
    constraints.append(x[:, 0] == x_init)
    constraints.append(z[0] == np.log(m0))
    constraints.append(z[N-1] >= np.log(m0 - mf))
    constraints.append(x[0, N-1] == 0)      # rx(tf) = 0
    constraints.append(x[3:6, N-1] == 0)    # v(tf) = 0
    
    # 2. Dynamics
    bg = B_cont @ g
    for k in range(N-1):
        constraints.append(x[:, k+1] == x[:, k] + dt * (A_cont @ x[:, k] + bg + B_cont @ u[:, k]))
        constraints.append(z[k+1] == z[k] - dt * alpha * gamma[k])

    # 3. Control Constraints
    constraints.append(cp.norm(u, axis=0) <= gamma)
    constraints.append(e1 @ u >= gamma * theta_cos)
    
    # 4. Thrust Bounds (Relaxed SOCP for Mass)
    z0_vec = np.array([np.log(m0 - alpha * rho2 * dt * i) for i in range(N)])
    lhs_factor = rho1 * np.exp(-z0_vec)
    rhs_factor = rho2 * np.exp(-z0_vec)
    delta_z = z - z0_vec
    constraints.append( cp.multiply(lhs_factor, (1 - delta_z)) <= gamma ) 
    constraints.append( gamma <= cp.multiply(rhs_factor, (1 - delta_z)) )
    
    # 5. State Constraints
    constraints.append(x[0, :] >= cp.norm(x[1:3, :], axis=0) * glide_tan)
    constraints.append(cp.norm(x[3:6, :], axis=0) <= velocity_max)
    
    return constraints

def run_cvxpy_solver(theta_deg, flight_time, x_init_val, target_q=np.array([0,0])):
    dt = 1.0
    N = int(flight_time / dt)
    theta_cos = np.cos(np.deg2rad(theta_deg))
    glide_tan = np.tan(np.deg2rad(glidelslope_angle))
    
    start_time = time.time()
    
    # --- Problem 3: Minimum Landing Error ---
    x = cp.Variable((6, N))
    z = cp.Variable(N)
    u = cp.Variable((3, N))
    gamma = cp.Variable(N)
    
    cons_p3 = cvx_get_constraints(x, z, u, gamma, N, dt, x_init_val, theta_cos, glide_tan)
    landing_pos = E_mat @ x[:, N-1]
    prob_p3 = cp.Problem(cp.Minimize(cp.norm(landing_pos - target_q)), cons_p3)
    
    try:
        prob_p3.solve(solver=cp.ECOS)
    except:
        prob_p3.solve(solver=cp.SCS)

    if prob_p3.status not in ["optimal", "optimal_inaccurate"]:
        return None, 0

    best_landing_dist = prob_p3.value
    
    # --- Problem 4: Minimum Fuel ---
    x2 = cp.Variable((6, N))
    z2 = cp.Variable(N)
    u2 = cp.Variable((3, N))
    gamma2 = cp.Variable(N)
    
    cons_p4 = cvx_get_constraints(x2, z2, u2, gamma2, N, dt, x_init_val, theta_cos, glide_tan)
    landing_pos2 = E_mat @ x2[:, N-1]
    cons_p4.append(cp.norm(landing_pos2 - target_q) <= best_landing_dist + 1.0)
    
    prob_p4 = cp.Problem(cp.Minimize(cp.sum(gamma2) * dt), cons_p4)
    try:
        prob_p4.solve(solver=cp.ECOS)
    except:
        prob_p4.solve(solver=cp.SCS)
        
    end_time = time.time()
    solve_duration = end_time - start_time
        
    if prob_p4.status not in ["optimal", "optimal_inaccurate"]:
        return None, solve_duration
        
    res = {
        'time': np.linspace(0, flight_time - dt, N),
        'pos': x2.value[:3, :],
        'vel': x2.value[3:6, :],
        'u': u2.value,
        'gamma': gamma2.value,
        'mass': np.exp(z2.value),
        'theta_deg': theta_deg,
        'tf': flight_time,
        'solve_time': solve_duration
    }
    return res, solve_duration

# --------------------------------------------------------------------------------
# PART 3: SCIPY SOLVER LOGIC
# --------------------------------------------------------------------------------

def unpack_vars(opt_vars, N):
    len_x = 6 * N
    len_z = N
    len_u = 3 * N
    x = opt_vars[0 : len_x].reshape((6, N))
    z = opt_vars[len_x : len_x + len_z]
    u = opt_vars[len_x + len_z : len_x + len_z + len_u].reshape((3, N))
    gamma = opt_vars[len_x + len_z + len_u : ]
    return x, z, u, gamma

def pack_vars(x, z, u, gamma):
    return np.concatenate([x.flatten(), z.flatten(), u.flatten(), gamma.flatten()])

def scipy_constraints_manager(opt_vars, N, dt, x_init, theta_cos, glide_tan, mode='error', max_error_val=None):
    x, z, u, gamma = unpack_vars(opt_vars, N)
    eq_cons, ineq_cons = [], []
    
    # Boundary Conditions
    eq_cons.append(x[:, 0] - x_init)
    eq_cons.append(z[0] - np.log(m0))
    eq_cons.append(x[0, N-1])
    eq_cons.append(x[3:6, N-1])
    
    # Dynamics
    bg = B_cont @ g 
    dx = (A_cont @ x[:, :-1]) + (B_cont @ u[:, :-1]) + bg[:, None]
    eq_cons.append((x[:, 1:] - (x[:, :-1] + dt * dx)).flatten())
    
    dz = -alpha * gamma[:-1]
    eq_cons.append((z[1:] - (z[:-1] + dt * dz)).flatten())

    # Inequalities
    ineq_cons.append(z[N-1] - np.log(m0 - mf)) # Fuel
    ineq_cons.append(gamma - np.linalg.norm(u, axis=0)) # Magnitude
    ineq_cons.append(u[0, :] - gamma * theta_cos) # Pointing
    ineq_cons.append(x[0, :] - glide_tan * np.linalg.norm(x[1:3, :], axis=0)) # Glideslope
    ineq_cons.append(velocity_max - np.linalg.norm(x[3:6, :], axis=0)) # Max Vel
    
    # Thrust Bounds (Approximation)
    z0_vec = np.array([np.log(m0 - alpha * rho2 * dt * i) for i in range(N)])
    term_common = 1 - (z - z0_vec)
    lhs = rho1 * np.exp(-z0_vec) * term_common
    rhs = rho2 * np.exp(-z0_vec) * term_common
    ineq_cons.append(gamma - lhs)
    ineq_cons.append(rhs - gamma)
    
    if mode == 'fuel' and max_error_val is not None:
        landing_pos = E_mat @ x[:, N-1]
        ineq_cons.append(max_error_val + 1.0 - np.linalg.norm(landing_pos))
        
    return np.concatenate([np.atleast_1d(c).flatten() for c in eq_cons]), \
           np.concatenate([np.atleast_1d(c).flatten() for c in ineq_cons])

def run_scipy_solver(theta_deg, flight_time, x_init_val, target_q=np.array([0,0])):
    dt = 1.0
    N = int(flight_time / dt)
    theta_cos = np.cos(np.deg2rad(theta_deg))
    glide_tan = np.tan(np.deg2rad(glidelslope_angle))
    
    start_time = time.time()

    # Initial Guess
    x_guess = np.zeros((6, N))
    for i in range(6): x_guess[i, :] = np.linspace(x_init_val[i], 0, N)
    z_guess = np.linspace(np.log(m0), np.log(m0-mf), N)
    u_guess = np.zeros((3, N))
    u_guess[0, :] = -g[0] * m0 * 0.5 
    gamma_guess = np.linalg.norm(u_guess, axis=0)
    x0_flat = pack_vars(x_guess, z_guess, u_guess, gamma_guess)

    # P3: Min Error
    cons_p3 = [
        {'type': 'eq',   'fun': lambda v: scipy_constraints_manager(v, N, dt, x_init_val, theta_cos, glide_tan, 'error')[0]},
        {'type': 'ineq', 'fun': lambda v: scipy_constraints_manager(v, N, dt, x_init_val, theta_cos, glide_tan, 'error')[1]}
    ]
    res_p3 = minimize(lambda v: np.linalg.norm(E_mat @ unpack_vars(v, N)[0][:, N-1] - target_q), 
                      x0_flat, method='SLSQP', constraints=cons_p3, 
                      options={'maxiter': 150, 'ftol': 1e-3, 'disp': False})
    
    best_landing_dist = res_p3.fun

    # P4: Min Fuel
    cons_p4 = [
        {'type': 'eq',   'fun': lambda v: scipy_constraints_manager(v, N, dt, x_init_val, theta_cos, glide_tan, 'fuel', best_landing_dist)[0]},
        {'type': 'ineq', 'fun': lambda v: scipy_constraints_manager(v, N, dt, x_init_val, theta_cos, glide_tan, 'fuel', best_landing_dist)[1]}
    ]
    res_p4 = minimize(lambda v: np.sum(unpack_vars(v, N)[3]) * dt, 
                      res_p3.x, method='SLSQP', constraints=cons_p4, 
                      options={'maxiter': 150, 'ftol': 1e-3, 'disp': False})
    
    end_time = time.time()
    solve_duration = end_time - start_time
    
    x_sol, z_sol, u_sol, gamma_sol = unpack_vars(res_p4.x, N)
    res = {
        'time': np.linspace(0, flight_time - dt, N),
        'pos': x_sol[:3, :],
        'vel': x_sol[3:6, :],
        'u': u_sol,
        'gamma': gamma_sol,
        'mass': np.exp(z_sol),
        'theta_deg': theta_deg,
        'tf': flight_time,
        'solve_time': solve_duration
    }
    return res, solve_duration

# --------------------------------------------------------------------------------
# PART 4: PLOTTING & EXECUTION
# --------------------------------------------------------------------------------

def plot_row(axes_row, results, title_prefix):
    """Helper to plot 3 subplots for a given set of results."""
    # 1. Attitude
    ax = axes_row[0]
    for res in results:
        u_vecs = res['u']
        u_norms = np.linalg.norm(u_vecs, axis=0)
        valid = u_norms > 1e-3
        angles = np.zeros_like(u_norms)
        cos_alpha = np.clip(u_vecs[0, valid] / u_norms[valid], -1.0, 1.0)
        angles[valid] = np.degrees(np.arccos(cos_alpha))
        ax.plot(res['time'], angles, label=res['label'], linewidth=2)
    ax.set_title(f"{title_prefix}: Attitude")
    ax.set_ylabel("Angle (deg)")
    ax.grid(True)
    ax.legend(loc='upper right', fontsize='small')

    # 2. Throttle
    ax = axes_row[1]
    for res in results:
        mass = res['mass']
        u_norms = np.linalg.norm(res['u'], axis=0)
        throttle_pct = ((u_norms * mass) / Tmax) * 100
        ax.plot(res['time'], throttle_pct, label=res['label'], linewidth=2)
    ax.axhline(y=(rho1/Tmax)*100, color='k', linestyle='--', alpha=0.3)
    ax.axhline(y=(rho2/Tmax)*100, color='k', linestyle='--', alpha=0.3)
    ax.set_title(f"{title_prefix}: Throttle %")
    ax.set_ylim(0, 100)
    ax.grid(True)

    # 3. Trajectory
    ax = axes_row[2]
    for res in results:
        ax.plot(res['pos'][1, :], res['pos'][2, :], label=res['label'], linewidth=2)
    ax.set_title(f"{title_prefix}: Y-Z Trajectory")
    ax.set_xlabel("Y (m)")
    ax.set_ylabel("Z (m)")
    ax.grid(True)

def print_table(solver_name, results):
    print(f"\nTABLE: {solver_name} Solver Results")
    print(f"{'Scenario':<25} | {'Fuel (kg)':<10} | {'Flight Time (s)':<15} | {'CPU Time (s)':<12}")
    print("-" * 70)
    for res in results:
        fuel = m0 - res['mass'][-1]
        print(f"{res['label']:<25} | {fuel:<10.1f} | {res['tf']:<15.2f} | {res['solve_time']:<12.4f}")
    print("=" * 70)

def main():
    x0_val = np.array([2400, 450, -330, -10, -40, 10])
    # x0_val = np.array([2400, 3400, 0, -40, 45, 0])
    scenarios = [
        {"label": "Unconstrained", "theta": 180, "tf": 45}, 
        {"label": "90 deg Limit", "theta": 90,  "tf": 47}, 
        {"label": "45 deg Limit", "theta": 45,  "tf": 58} 
    ]
    
    res_cvx = []
    res_sci = []
    
    # Run CVXPY
    print("Running CVXPY Solver...")
    for s in scenarios:
        res, _ = run_cvxpy_solver(s['theta'], s['tf'], x0_val)
        if res:
            res['label'] = s['label']
            res_cvx.append(res)
            
    # Run SciPy
    print("Running SciPy Solver...")
    for s in scenarios:
        res, _ = run_scipy_solver(s['theta'], s['tf'], x0_val)
        if res:
            res['label'] = s['label']
            res_sci.append(res)
            
    # Generate Plots
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    plot_row(axes[0], res_cvx, "CVXPY")
    plot_row(axes[1], res_sci, "SciPy")
    
    plt.tight_layout()
    plt.savefig('combined_solver_comparison.png')
    print("\nPlot saved as 'combined_solver_comparison.png'")
    
    # Print Tables
    print_table("CVXPY", res_cvx)
    print_table("SciPy (SLSQP)", res_sci)

if __name__ == "__main__":
    main()