import numpy as np

def make_toy_batch(num_scenes=2, agents_per_scene=5, T_past=8, T_future=12, seed=0):
    """
    Returns dict with:
      past:   (S, A, T_past, 2)
      future: (S, A, T_future, 2)
    """
    rng = np.random.default_rng(seed)
    past, future = [], []
    for _ in range(num_scenes):
        centers = rng.uniform(-10, 10, size=(agents_per_scene, 2))
        headings = rng.uniform(-np.pi, np.pi, size=(agents_per_scene,))
        v = rng.uniform(0.5, 2.0, size=(agents_per_scene,))
        p, f = [], []
        for a in range(agents_per_scene):
            vel = v[a]*np.array([np.cos(headings[a]), np.sin(headings[a])])
            t_past = np.arange(-T_past+1, 1)[:, None]
            traj_p = centers[a] + t_past*vel + rng.normal(scale=0.05, size=(T_past, 2))
            p.append(traj_p)
            t_future = np.arange(1, T_future+1)[:, None]
            traj_f = centers[a] + t_future*vel + rng.normal(scale=0.05, size=(T_future, 2))
            f.append(traj_f)
        past.append(np.stack(p, axis=0))
        future.append(np.stack(f, axis=0))
    return {"past": np.stack(past), "future": np.stack(future)}
