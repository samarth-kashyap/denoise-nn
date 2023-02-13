import numpy as np
import matplotlib.pyplot as plt
import argparse


def background(x, a, b):
    xhalfidx = np.argmin(abs(x-x[len(x)//2]))
    xhalf = x[xhalfidx]
    return a - b*((x - xhalf)/10)**2
    

def lorentzian(x, x0, gamma):
    return gamma/2/np.pi/((x-x0)**2 + gamma*gamma/4)



def get_realization(sig, max_scale=20., noise_scale=1.0, noisetype='chisq'):
    max_noise = sig.max()/max_scale

    if noisetype == 'gaussian':
        noisy_sig = sig + np.random.randn(len(sig))*sig/noise_scale

    elif noisetype == 'chisq':
        n1 = np.random.randn(len(sig))/noise_scale
        n2 = np.random.randn(len(sig))/noise_scale
        chisq2 = n1*n1 + n2*n2
        # noisy_sig = sig * ((1 + np.random.randn(len(sig))/noise_scale) *
                           # (1 + np.random.randn(len(sig))/noise_scale))
        noisy_sig = sig * chisq2

    return noisy_sig




x = np.load('data-files/x.npy')
x0_superlist = np.load('data-files/x0.npy')
gamma_superlist = np.load('data-files/gamma.npy')
bg_dcshift_superlist = np.load('data-files/bg_dcshift.npy')
bg_curvature_superlist = np.load('data-files/bg_curvature.npy')

idx = np.random.randint(len(x0_superlist))
x0list = x0_superlist[idx]
gammalist = gamma_superlist[idx]
gammalist2 = []

for i in range(len(gammalist)):
    gammalist2.append(0.03)
sig = 0
for idx, x0 in enumerate(x0list):
    gamma = gammalist2[idx]
    sig += lorentzian(x, x0, gamma)
bg_dcshift = np.random.uniform(low=0.8, high=0.95)
bg_curvature = np.random.uniform(low=0.05, high=0.25)
bg = background(x, bg_dcshift, bg_curvature)
sig += bg

realizations = []
for i in range(100):
    realizations.append(get_realization(sig))

meansig = sum(realizations)/100.
plt.figure()
for i in range(15):
    plt.plot(x, realizations[i], '.b', alpha=0.2)
plt.plot(x, meansig - bg, 'k')
plt.plot(x, sig - bg, 'r')
plt.show()

plt.figure()
for i in range(5):
    plt.plot(x, realizations[i]/meansig, '.b', alpha=0.7)
plt.show()
