

import numpy as np

def predict_cv(past, T_future):
    """
    constant velocity baseline
    past: (A, T_past, 2) → (A, T_future, 2)
    """
    vel = past[:, -1, :] - past[:, -2, :]
    fut = [past[:, -1, :] + (i+1)*vel for i in range(T_future)]
    return np.stack(fut, axis=1)
