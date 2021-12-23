import numpy as np 
from sklearn.ensemble import RandomForestRegressor 
from sklearn.metrics import mean_squared_error
import time


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

def experiment():
    results2 = np.zeros((9, 12))
    for j, d in enumerate(range(5, 14)):
        for k, n in enumerate(range(5, 17)):
            x_train, x_test = np.random.uniform(-1,1, size=(2**n, d)), np.random.uniform(-1,1, size=(2**n, d))

            bits = [g_gen(d) for i in range(10)]
            
            y_train = [[f_star(train, g) for train in x_train] for g in bits]
            y_test = [[f_star(test, g) for test in x_test] for g in bits]
            y_train = np.array(y_train).T
            y_test = np.array(y_test).T
            k_runs = []
            t1 = time.time()
            for i, g in enumerate(bits):
                model = RandomForestRegressor(n_jobs = -1,)
                y_hat = model.fit(x_train, y_train)
                y_pred = model.predict(x_test)
                mse = mean_squared_error(y_test, y_pred)
                mean_samp = 1/(2**n) * np.sum(y_pred)
                sd_samp = (1/(2**n-1)*np.sum((y_pred - mean_samp)**2))**(1/2)
                mse_sd = mse/sd_samp
                k_runs.append(mse_sd)
                
                print(f"Model Dimension: {d} \nModel Sample Size: {2**n}\nModel Number: {k}")

            t2 = time.time()
            print(f"Time take: {t2-t1}") 
            res = np.mean(k_runs)
            print(f"Average MSE Thing : {res}")
            results2[j, k] = res


if __name__ == "__main__":
    experiment()