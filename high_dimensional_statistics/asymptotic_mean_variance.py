import numpy as np
import matplotlib.pyplot as plt


def plot():
    n = 1000 
    D = [10, 100, 1000, 10000]
    var = []
    mean = []

    for d in D:
        samples = np.random.normal(scale=1/np.sqrt(d), size=(n, d))
        samples = np.power(samples,2)
        vec = np.ones(d) 
        l2 = samples @ vec 
        sample_mean = np.sum(l2)*1/n
        sample_var = np.sum((l2-sample_mean)**2)*1/(n-1)
        var.append(sample_var)
        mean.append(sample_mean)
        print(f"Sample Mean: {sample_mean}\n Sample SD: {np.sqrt(sample_var)}\n 1/sqrt(d): {1/np.sqrt(d)} ")
        #plt.plot(d, sample_var)
        #plt.plot(d, sample_var)

    plt.plot(D, var)
    plt.xlim(8)
    plt.title("Variance by Dimensions")
    plt.xlabel("Dimensions")
    plt.ylabel("Sample Variance")
    plt.show()
    plt.savefig("Variance.png")
    plt.title("Mean by Dimensions")
    plt.xlabel("Dimensions")
    plt.ylabel("Sample Mean")
    plt.plot(D, mean)
    plt.show()
    plt.savefig("Mean.png")


if __name__ == "__main__":
    plot()



    

