import numpy as np
from params import params


num_bins = params.num_bins
num_inputs = params.num_inputs
num_lorentzians = params.num_lorentzians
x = np.linspace(0, 7, params.num_bins)
interval_size = (x.max() - x.min())
xlo_start = x.min() + 0.2*interval_size
xhi_end = x.max() - 0.2*interval_size
reduced_interval = xhi_end - xlo_start

xlo = xlo_start + [reduced_interval/num_lorentzians*(i) for i in range(num_lorentzians)]
xhi = xlo_start + [reduced_interval/num_lorentzians*(1+i) for i in range(num_lorentzians)]

x0_superlist = []
gamma_superlist = []
bg_dcshift_superlist = []
bg_curvature_superlist = []

for i in range(num_inputs):
    x0list = [np.random.uniform(low=xlo[j], high=xhi[j])
              for j in range(num_lorentzians)]
    print(x0list)

    gammalist = [np.random.uniform(low=0.1, high=0.5)
                 for j in range(num_lorentzians)]

    bg_dcshift = np.random.uniform(low=0.8, high=0.95)
    bg_curvature = np.random.uniform(low=0.05, high=0.25)

    x0_superlist.append(x0list)
    gamma_superlist.append(gammalist)
    bg_dcshift_superlist.append(bg_dcshift)
    bg_curvature_superlist.append(bg_curvature)


np.save(f"data-files/x.npy", x)
np.save(f"data-files/x0.npy", x0_superlist)
np.save(f"data-files/gamma.npy", gamma_superlist)
np.save(f"data-files/bg_dcshift.npy", bg_dcshift_superlist)
np.save(f"data-files/bg_curvature.npy", bg_curvature_superlist)


