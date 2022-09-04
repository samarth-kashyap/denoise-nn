import numpy as np

x = np.linspace(0, 15, 1000)

num_inputs = 100

x0_superlist = []
gamma_superlist = []
bg_dcshift_superlist = []
bg_curvature_superlist = []

for i in range(100):
    x0list = [np.random.uniform(low=4, high=5), 
              np.random.uniform(low=5.2, high=6.2),
              np.random.uniform(low=6.3, high=7.3),
              np.random.uniform(low=7.4, high=8.5)]

    gammalist = [np.random.uniform(low=0.1, high=0.5),
                 np.random.uniform(low=0.1, high=0.5),
                 np.random.uniform(low=0.1, high=0.5),
                 np.random.uniform(low=0.1, high=0.5)]

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


