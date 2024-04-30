import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from sympy import symbols, solve, Matrix, N
from scipy.linalg import eigvals

alpha = 1.636
beta = 0.002
delta1 = 0.3743
w1 = 0.04
sigma2 = 0.38
delta2 = 0.055
xi = 0.05 

times = np.linspace(0, 800, 10000)

def noTrt():
    sigma1 = 0
    rho = 0.01
    w2 = 0.002
    init = [1, 1, 1]
    S = odeint(model, init, times, args=(sigma1, rho, w2))
    plt.plot(times, S[:,0], label='TC', linewidth=2)
    plt.plot(times, S[:,1], label='EC', linewidth=2)
    plt.plot(times, S[:,2], label='HTC', linewidth=2)
    plt.title('Original without Treatment')
    plt.ylabel("Concentration")
    plt.xlabel("Time")
    plt.legend()

def withTrt():
    times = np.linspace(0, 250, 10000)
    sigma1 = 0.1181
    rho = 0.01
    w2 = 0.002
    init = [1, 1, 1]
    S = odeint(model, init, times, args=(sigma1, rho, w2))
    plt.plot(times, S[:,0], label='TC', linewidth=2)
    plt.plot(times, S[:,1], label='EC', linewidth=2)
    plt.plot(times, S[:,2], label='HTC', linewidth=2)
    plt.title('Original with Treatment')
    plt.ylabel("Concentration")
    plt.xlabel("Time")
    plt.legend()

def withTrtOsc():
    sigma1 = 0.1181
    rho = 0.001
    w2 = 0.02
    init = [1, 1, 1]
    S = odeint(model, init, times, args=(sigma1, rho, w2))
    plt.yscale('log')
    plt.plot(times, S[:,0], label='TC', linewidth=2)
    plt.plot(times, S[:,1], label='EC', linewidth=2)
    plt.plot(times, S[:,2], label='HTC', linewidth=2)
    plt.title('Original with Treatment and Oscillation')
    plt.ylabel("Concentration")
    plt.xlabel("Time")
    plt.legend()

def noTrtiNKT():
    sigma1 = 0
    rho = 0.01
    w2 = 0.002
    sigma3 = 0.01
    w3 = 0.001
    delta3 = 0.055

    gamma = 1

    init = [1, 1, 1, .1]
    S = odeint(Nmodel, init, times, args=(sigma1, rho, w2, sigma3, w3, delta3, gamma))
    plt.plot(times, S[:,0], label='TC', linewidth=2)
    plt.plot(times, S[:,1], label='EC', linewidth=2)
    plt.plot(times, S[:,2], label='HTC', linewidth=2)
    plt.plot(times, S[:,3], label='iNKT', linewidth=2)
    plt.title('iNKT without Treatment')
    plt.ylabel("Concentration")
    plt.xlabel("Time")
    plt.legend()

    last_values = S[-1]
    return last_values


def withTrtiNKT():
    times = np.linspace(0, 250, 10000)
    sigma1 = 0.1181
    rho = 0.01
    w2 = 0.002
    sigma3 = 0.01
    w3 = 0.001
    delta3 = 0.055
    gamma = 1.6
    init = [1, 1, 1, .1]
    S = odeint(Nmodel, init, times, args=(sigma1, rho, w2, sigma3, w3, delta3, gamma))
    #plt.yscale('log')
    plt.plot(times, S[:,0], label='TC', linewidth=2)
    plt.plot(times, S[:,1], label='EC', linewidth=2)
    plt.plot(times, S[:,2], label='HTC', linewidth=2)
    plt.plot(times, S[:,3], label='iNKT', linewidth=2)
    plt.title('iNKT Interaction with Treatment')
    plt.ylabel("Concentration")
    plt.xlabel("Time")
    plt.legend()

    last_values = S[-1]
    return last_values
    
def withTrtOsciNKT():
    sigma1 = 0.1181
    rho = 0.001
    w2 = 0.02
    w3 = 0.01
    sigma3 = 0.01
    delta3 = 0.055
    gamma = 0.05
    init = [1, 1, 1, .1]
    S = odeint(Nmodel, init, times, args=(sigma1, rho, w2, sigma3, w3, delta3, gamma))
    plt.yscale('log')
    plt.plot(times, S[:,0], label='TC', linewidth=2)
    plt.plot(times, S[:,1], label='EC', linewidth=2)
    plt.plot(times, S[:,2], label='HTC', linewidth=2)
    plt.plot(times, S[:,3], label='iNKT', linewidth=2)
    plt.title('iNKT with Treatment and Oscillation')
    plt.ylabel("Concentration")
    plt.xlabel("Time")
    plt.legend()

    last_values = S[-1]
    return last_values
    
def model(S, t, sigma1, rho, w2):
    dS = np.zeros(3)
    x, y, z = S
    dS[0] = alpha*x*(1 - beta*x) - x*y              # TC
    dS[1] = sigma1 + w1*x*y - delta1*y + rho*y*z    # EC CD8+
    dS[2] = sigma2 + w2*x*z - delta2*z              # HTC CD4+
    return dS


def Nmodel(S, t, sigma1, rho, w2, sigma3, w3, delta3, gamma):
    dS = np.zeros(4)
    x, y, z, iNKT = S
    dS[0] = alpha*x*(1 - beta*x) - x*y - x*xi*iNKT                  # TC
    dS[1] = sigma1 + w1*x*y - delta1*y + rho*y*z + gamma*y*iNKT     # EC CD8+
    dS[2] = sigma2 + w2*x*z - delta2*z                              # HTC CD4+
    dS[3] = sigma3 + w3*x*iNKT - delta3*iNKT                        # iNKT
    return dS


fig, axs = plt.subplots(1, 2)
plt.sca(axs[0])
withTrtOsc()
plt.sca(axs[1])
withTrtOsciNKT()
plt.tight_layout()
plt.show()


def eigenvalue_cal(TC, EC, HTC, iNKT):
    alpha = 1.636
    beta = 0.002
    delta1 = 0.3743
    w1 = 0.04
    sigma2 = 0.38
    delta2 = 0.055
    sigma1 = 0.1181
    rho = 0.01
    w2 = 0.002
    sigma3 = 0.01
    w3 = 0.001
    delta3 = 0.055
    gamma = 0.9
    
    J_evaluated = Matrix([
        [alpha * (1 - 2 * beta * TC) - EC - xi * iNKT, -TC, 0, -TC * xi],
        [EC * w1, w1 * TC - delta1 + rho * HTC + gamma * iNKT, EC * rho, EC * gamma],
        [w2 * HTC, 0, w2 * TC - delta2, 0],
        [w3 * iNKT, 0, 0, w3 * TC - delta3]
    ])

    eigenvalues_evaluated = J_evaluated.eigenvals()
    print(eigenvalues_evaluated)
