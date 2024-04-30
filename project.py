import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import fsolve

alpha = 1.636
beta = 0.002
delta1 = 0.3743
w1 = 0.04
sigma2 = 0.38
delta2 = 0.055
times = np.linspace(0, 800, 10000)

def run(sigma1, rho, w2, opt=True):
    times = np.linspace(0, 800, 10000)
    trt = 'with Treatment' if sigma1 !=0 else 'no Treatment'
    main = trt+': $\\rho=$'+str(rho)+', $\omega_2=$'+str(w2)
    init = [1, 1, 1]
    S = odeint(model, init, times, args=(sigma1, rho, w2))
    if opt:
        plt.plot(times, S[:,0], label='TC', linewidth=2)
        plt.plot(times, S[:,1], label='EC', linewidth=2)
        plt.plot(times, S[:,2], label='HTC', linewidth=2)
        plt.title(main)
        plt.ylabel("Concentration")
        plt.xlabel("Time")
        plt.legend()
    else:
        zline = S[:,0]
        xline = S[:,1]
        yline = S[:,2]
        axs.plot3D(xline, yline, zline, 'gray')
        axs.view_init(-160, 160)
        axs.set_xlabel('y', labelpad=20)
        axs.set_ylabel('z', labelpad=20)
        axs.set_zlabel('x', labelpad=20)
        axs.set_title('Surface Plot of '+main)

def model(S, t, sigma1, rho, w2):
    dS = np.zeros(3)
    x, y, z = S
    dS[0] = alpha*x*(1 - beta*x) - x*y              # TC
    dS[1] = sigma1 + w1*x*y - delta1*y + rho*y*z    # EC CD8+
    dS[2] = sigma2 + w2*x*z - delta2*z              # HTC CD4+
    return dS

plt.figure()
run(0, 0.01, 0.002)
plt.show()

fig, axs = plt.subplots(1,2)
plt.sca(axs[0])
plt.yscale('log')
run(0.1181, 0.001, 0.015)
plt.sca(axs[1])
plt.yscale('log')
run(0.1181, 0.042, 0.015)
plt.tight_layout()
plt.show()

fig, axs = plt.subplots(1,2)
plt.sca(axs[0])
run(0.1181, 0.05, 0.015)
plt.sca(axs[1])
run(0.1181, 0.055, 0.015)
plt.tight_layout()
plt.show()

fig = plt.figure()
axs = fig.add_subplot(1, 2, 1, projection='3d')
run(0.1181, 0.015, 0.015, False)
axs = fig.add_subplot(1, 2, 2, projection='3d')
run(0.1181, 0.001, 0.02, False)
plt.show()