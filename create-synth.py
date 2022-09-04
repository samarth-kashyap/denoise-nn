import numpy as np
import matplotlib.pyplot as plt
import argparse



def background(x, a, b):
    xhalfidx = np.argmin(abs(x-x[len(x)//2]))
    xhalf = x[xhalfidx]
    return a - b*((x - xhalf)/10)**2
    

def lorentzian(x, x0, gamma):
    return gamma/2/np.pi/((x-x0)**2 + gamma*gamma/4)


def get_realization(sig, max_scale=20., noise_scale=3.):
    max_noise = sig.max()/max_scale
    noisy_sig = sig + np.random.randn(len(sig))*sig/noise_scale
    return noisy_sig


x = np.linspace(0, 15, 1000)
x0list = [np.random.uniform(low=4, high=5), 
          np.random.uniform(low=5.2, high=6.2),
          np.random.uniform(low=6.3, high=7.3),
          np.random.uniform(low=7.4, high=8.5)]

gammalist = [np.random.uniform(low=0.1, high=0.5),
             np.random.uniform(low=0.1, high=0.5),
	     np.random.uniform(low=0.1, high=0.5),
             np.random.uniform(low=0.1, high=0.5)]

sig = 0
for idx, x0 in enumerate(x0list):
    gamma = gammalist[idx]
    sig += lorentzian(x, x0, gamma)
bg_dcshift = np.random.uniform(low=0.8, high=0.95)
bg_curvature = np.random.uniform(low=0.05, high=0.25)
bg = background(x, bg_dcshift, bg_curvature)
sig += bg

realizations = []
for i in range(100):
    realizations.append(get_realization(sig))

plt.figure()
for i in range(5):
    plt.plot(x, realizations[i], '.b', alpha=0.7)
plt.plot(x, sum(realizations)/100., 'k')
plt.plot(x, sig - bg, 'r')
plt.show()
