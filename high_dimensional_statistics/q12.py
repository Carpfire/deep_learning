import numpy as np 
import matplotlib.pyplot as plt



def psi(x):

    dist = np.abs(x) - .5
    ind = np.argmin(np.abs(dist))
    
    y = np.abs(dist[ind])
    return y, ind

def g_gen(size):
    n = 2**size
    return np.random.randint(2, size = n) *2 -1



def f_star(x, g):
    if np.any(x == 0):
        return 0

    else:
        
        z_use = np.sign(x)
        x_use = x - z_use/2
        #print(x_use)
        P, ind = psi(x_use)
        g_use = g[ind]
        return g_use*P


def plot():
    xs, ys = np.linspace(-1, 1, 10), np.linspace(-1, 1, 10)
    g = g_gen(2)
    X, Y = np.meshgrid(xs, ys)

    coords = np.vstack((X.ravel(), Y.ravel())).T
    z = np.array([f_star(x,g) for x in coords])
    z = z.reshape(X.shape)
    plt.contourf(xs, ys, z)
    plt.title("2-D plot of Psi")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()
    plt.savefig("Pretty_Pic.png")


if __name__ == "__main__":
    plot()

