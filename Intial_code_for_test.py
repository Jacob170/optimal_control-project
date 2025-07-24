import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# ------------------------------------------------------------------
# 6.1  Core Functions  –  Problem constants  (SI units)
# ------------------------------------------------------------------
V     = 3000 * 0.3048         # closing speed [m s⁻¹]
tf    = 10.0                  # final time [s]
tau   = 2.0                   # correlation time of a_T [s]
R1    = 15e-6                 # measurement-noise base [rad² s⁻¹]
R2    = 1.67e-3               # measurement-noise scaling [rad² s³]
W     = 100.0                 # process-noise intensity  [(m s⁻²)² s]
b     = 1.52e-2               # control-effort weight
P22_0 = 16.0                  # var(v₀)  [(m s⁻¹)²]
P33_0 = 400.0                 # var(a_T(0))  [(m s⁻²)²]

def _dtg(t):
    return max(tf - t, 1e-6)

def Rm(t):
    dtg = _dtg(t)
    return R1 + R2 / dtg**2

def H(t):
    dtg = _dtg(t)
    return np.array([[1.0 / (V * dtg), 0.0, 0.0]])

def state_matrices():
    A = np.array([[0, 1, 0],
                  [0, 0, 1],
                  [0, 0, -1/tau]])
    B = np.array([[0],
                  [1],
                  [0]])
    G = np.array([[0],
                  [0],
                  [np.sqrt(W)]])
    return A, B, G

def filter_riccati_rhs(t, P_flat):
    A, _, G = state_matrices()
    P = P_flat.reshape(3, 3)
    H_t = H(t)
    inv_Rm = 1.0 / Rm(t)
    dP = A @ P + P @ A.T - P @ H_t.T * inv_Rm @ H_t @ P + G @ G.T
    return dP.flatten()

def control_riccati_rhs(t, S_flat):
    A, B, _ = state_matrices()
    S = S_flat.reshape(3, 3)
    dS = -(A.T @ S + S @ A - (1.0 / b) * (S @ B @ B.T @ S))
    return dS.flatten()

def simulate_closed_loop(seed, time_grid, S_traj, P_traj):
    np.random.seed(seed)
    dt = time_grid[1] - time_grid[0]
    A, B, G = state_matrices()

    x = np.array([0.0,
                  np.random.normal(0, np.sqrt(P22_0)),
                  np.random.normal(0, np.sqrt(P33_0))])
    x_hat = np.zeros(3)
    y_hist, u_hist = [], []

    for i, t in enumerate(time_grid):
        S = S_traj[i].reshape(3, 3)
        P = P_traj[i].reshape(3, 3)
        H_t = H(t)
        inv_Rm = 1.0 / Rm(t)

        K = (1.0 / b) * (B.T @ S)
        L = P @ H_t.T * inv_Rm

        z = H_t @ x + np.random.normal(0, np.sqrt(Rm(t)))
        u = -K @ x_hat

        w = np.random.normal(0, 1)
        x_dot = A @ x + B.flatten() * u + G.flatten() * w
        x += x_dot * dt

        innovation = z - H_t @ x_hat
        x_hat_dot = A @ x_hat + B.flatten() * u + L.flatten() * innovation
        x_hat += x_hat_dot * dt

        y_hist.append(x[0])
        u_hist.append(u)

    y_tf = x[0]
    J = 0.5 * y_tf**2 + 0.5 * b * np.sum(np.square(u_hist)) * dt
    return y_tf, J, y_hist, u_hist

def run_monte_carlo(N=1000, time_steps=200, sol_S=None, sol_P=None):
    time_grid = np.linspace(0, tf, time_steps)
    P_traj = sol_P.y.T
    S_traj = sol_S.y.T[::-1]
    y_finals, costs = [], []
    y_histories = []
    u_histories = []

    for i in range(N):
        y_tf, J, y_hist, u_hist = simulate_closed_loop(i, time_grid, S_traj, P_traj)
        y_finals.append(y_tf)
        costs.append(J)
        y_histories.append(y_hist)
        u_histories.append(u_hist)

    return np.array(y_finals), np.array(costs), np.array(y_histories), np.array(u_histories), time_grid

def plot_results(y_all, J_all, y_histories, u_histories, time_grid):
    print(f"Final Statistics (N = {len(y_all)} simulations):")
    print(f"Mean y(tf):      {np.mean(y_all):.4f} m")
    print(f"Std  y(tf):      {np.std(y_all):.4f} m")
    print(f"Mean cost J:     {np.mean(J_all):.4f}")
    print(f"Std  cost J:     {np.std(J_all):.4f}")

    # Figure 1: Mean ± 1σ of y(t)
    y_mean = np.mean(y_histories, axis=0)
    y_std = np.std(y_histories, axis=0)
    plt.figure()
    plt.plot(time_grid, y_mean, label="Mean y(t)", color="blue")
    plt.fill_between(time_grid, y_mean - y_std, y_mean + y_std, alpha=0.3, label="±1σ", color="blue")
    plt.title("Figure 1: Mean ± 1σ of y(t)")
    plt.xlabel("Time [s]")
    plt.ylabel("y(t) [m]")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("figure1.png")
    plt.show()

    # Figure 2: Representative control history
    plt.figure()
    plt.plot(time_grid, u_histories[0], label="a_P(t)", color="green")
    plt.title("Figure 2: Representative Control History a_P(t)")
    plt.xlabel("Time [s]")
    plt.ylabel("Control [m/s²]")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("figure2.png")
    plt.show()

    # Figure 3: Histogram of Final Miss Distance
    plt.figure(figsize=(10, 4))
    plt.hist(y_all, bins=40, alpha=0.7, color='royalblue', edgecolor='black')
    plt.title("Figure 3: Histogram of Final Miss Distance y(tf)")
    plt.xlabel("y(tf) [m]")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("figure3.png")
    plt.show()

    # Figure 4: Histogram of Total Cost
    plt.figure(figsize=(10, 4))
    plt.hist(J_all, bins=40, alpha=0.7, color='darkorange', edgecolor='black')
    plt.title("Figure 4: Histogram of Total Cost J")
    plt.xlabel("Cost J")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("figure4.png")
    plt.show()


def is_positive_semidefinite(matrix):
    eigvals = np.linalg.eigvalsh(matrix)
    return np.all(eigvals >= -1e-10)

if __name__ == "__main__":
    print("Validating Riccati solutions:")

    t_grid = np.linspace(0, tf, 200)
    P0 = np.diag([0, P22_0, P33_0])
    Sf = np.diag([1, 0, 0])

    sol_P = solve_ivp(filter_riccati_rhs, [0, tf], P0.flatten(), t_eval=t_grid,
                      rtol=1e-8, atol=1e-10)
    sol_S = solve_ivp(control_riccati_rhs, [tf, 0], Sf.flatten(), t_eval=t_grid[::-1],
                      rtol=1e-8, atol=1e-10)

    valid_P = all(is_positive_semidefinite(P.reshape(3, 3)) for P in sol_P.y.T)
    valid_S = all(is_positive_semidefinite(S.reshape(3, 3)) for S in sol_S.y.T)

    print(f"P(t) is positive semi-definite for all t? {valid_P}")
    print(f"S(t) is positive semi-definite for all t? {valid_S}")

    y_all, J_all, y_histories, u_histories, t_grid = run_monte_carlo(N=1000, sol_P=sol_P, sol_S=sol_S)
    print("Monte Carlo complete.")
    plot_results(y_all, J_all, y_histories, u_histories, t_grid)

    print("Ricatti ODEs solved without errors:", sol_P.success and sol_S.success)

