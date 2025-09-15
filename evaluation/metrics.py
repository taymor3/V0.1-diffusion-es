import numpy as np

def rmse(pred, gt):
    return float(np.sqrt(np.mean((pred - gt)**2)))

def count_collisions(traj, radius=0.5):
    A, T, _ = traj.shape
    collisions = 0
    for t in range(T):
        dists = np.linalg.norm(traj[:,None,t,:] - traj[None,:,t,:], axis=-1)
        close = (dists < radius) & (~np.eye(A, dtype=bool))
        if np.any(close):
            collisions += 1
    return collisions
