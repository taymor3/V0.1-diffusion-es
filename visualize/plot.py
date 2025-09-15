import matplotlib.pyplot as plt

def plot_scene(past, future, title="Toy scene"):
    plt.figure()
    for a in range(past.shape[0]):
        plt.plot(past[a,:,0], past[a,:,1], "--", marker=".")
        plt.plot(future[a,:,0], future[a,:,1], "-", marker="o")
    plt.gca().set_aspect("equal")
    plt.title(title)
    plt.grid(True)
    plt.show()
