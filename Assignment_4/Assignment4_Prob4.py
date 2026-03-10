import numpy as np
import matplotlib.pyplot as plt
import os
os.makedirs("output", exist_ok=True)
np.random.seed(42)
def sample_from_multivariate_normal(mean,cov):
    return np.random.multivariate_normal(mean,cov, 1).T

def kalman_filter(A,B,C,W,V,X_0,K, emp_cov = False):
    #Generate Data
    x_0 = sample_from_multivariate_normal(np.zeros_like(X_0[:,0]), X_0)
    x_data = []
    y_data = []
    for k in range(K):
        w_k = sample_from_multivariate_normal(np.zeros_like(W[:,0]), W)
        v_k = sample_from_multivariate_normal([0], V)
        x_k = A @ x_0 + B @ w_k
        y_k = C @ x_k + v_k
        x_0 = x_k
        x_data.append(x_k)
        y_data.append(y_k)
    
    x_k_k1_track = []
    x_k_k_track = []
    R_k_k1_track = []

    #Prediction and Update
    for k in range(K):
        #Intialization
        if k == 0:
            x_k_k1 = np.zeros((A.shape[0], 1))
            R_k_k1 = X_0

        #Kalman Gain
        L_k = R_k_k1 @ C.T @ np.linalg.inv(C @ R_k_k1 @ C.T + V)

        #Update
        R_k_k = R_k_k1 - L_k @ C @ R_k_k1
        x_k_k = x_k_k1 + L_k @ (y_data[k] - C @ x_k_k1)

        x_k_k1_track.append(x_k_k1)
        x_k_k_track.append(x_k_k)
        R_k_k1_track.append(R_k_k1)

        #Prediction
        R_k_k1 = A @ R_k_k @ A.T + B @ W @ B.T
        x_k_k1 = A @ x_k_k

    x_data = np.array(x_data)
    x_k_k_track = np.array(x_k_k_track)
    x_k_k1_track = np.array(x_k_k1_track)
    trace_theoretical = np.array([np.trace(R) for R in R_k_k1_track])

    if not emp_cov:
        v_val = V[0,0]
        _, ax = plt.subplots(figsize=(10, 4))
        ax.plot(np.arange(K), x_data[:, 0, 0],
                label=r'$x_k$ (true)', color='steelblue', linewidth=1.2)
        ax.plot(np.arange(K), x_k_k_track[:, 0, 0],
                label=r'$\hat{x}_{k|k}$ (KF estimate)', color='tomato',
                linewidth=1.2, linestyle='--')
        ax.set_xlabel('Time step $k$')
        ax.set_ylabel('State $x_1$')
        ax.set_title(rf'Kalman Filter: True State vs Filtered Estimate  ($V = {v_val}$)')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'output/KF_trajectory_v={v_val}.png', dpi=150, bbox_inches='tight')
        plt.show()

    return x_data, x_k_k_track, x_k_k1_track, trace_theoretical

def empirical_covariance(A,B,C,W,V,X_0,K, M=100000):
    n = A.shape[0]
    S_k = np.zeros((K, n, n))
    trace_theoretical = None
    for i in range(M):
        xk, _, x_kk1, trace_th = kalman_filter(A,B,C,W,V,X_0,K, emp_cov=True)
        error = xk - x_kk1                         # (K, n, 1) prediction error
        S_k += error @ error.transpose(0, 2, 1)    # (K, n, n)
        if trace_theoretical is None:
            trace_theoretical = trace_th
        
        if i%1000 == 0:
            print(f'Iteration: {i}')

    trace_empirical = np.einsum('kii->k', S_k) / M

    v_val = V[0, 0]
    _, ax = plt.subplots(figsize=(10, 4))
    ax.plot(np.arange(K), trace_theoretical,
            label=r'Theoretical $\mathrm{tr}(R_{k|k})$', color='steelblue', linewidth=1.2)
    ax.plot(np.arange(K), trace_empirical,
            label=r'Empirical $\mathrm{tr}(\hat{R}_{k|k})$', color='tomato',
            linewidth=1.2, linestyle='--')
    ax.set_xlabel('Time step $k$')
    ax.set_ylabel(r'$\mathrm{tr}(R_{k|k})$')
    ax.set_title(rf'KF Error Covariance: Theoretical vs Empirical  ($V = {v_val},\, M = {M}$)')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'output/KF_covariance_v={v_val}_M={M}.png', dpi=150, bbox_inches='tight')
    plt.show()

    return trace_empirical

if __name__ == "__main__":
    A = np.array([[0.8, -0.25, 0],[-0.8, -0.1,0],[0,-0.5,0.4]])
    B = np.array([[1,0],[0,1],[0,0]])
    C = np.array([[1,2,3]])
    W = np.array([[2,-1],[-1,3]])
    # V = np.array([[1]])
    X_0 = np.eye(3)
    K = 100
    k_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    # Part (b): trajectory plots for each V
    for k in k_list:
        V = np.array([[k]])
        kalman_filter(A,B,C,W,V,X_0,K)

    # Part (c): theoretical vs empirical covariance for V=1
    V = np.array([[1]])
    empirical_covariance(A,B,C,W,V,X_0,K)

