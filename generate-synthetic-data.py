import numpy as np
import matplotlib.pyplot as plt
import argparse
from datetime import date
from datetime import datetime
from tqdm import tqdm
import functions as fn
from params import params
import os

training_or_testing = 'train'

todays_date = date.today()
timeprefix = datetime.now().strftime("%H.%M")
dateprefix = f"{todays_date.day:02d}.{todays_date.month:02d}.{todays_date.year:04d}"
timestamp = f"{dateprefix}-{timeprefix}"
store_dir = f'{training_or_testing}-set'
print(f"store directory = {store_dir}")
try:
    os.system(f"mkdir {store_dir}/noisy")
    os.system(f"mkdir {store_dir}/target")
    os.system(f"mkdir {store_dir}/torch")
except:
    pass

try:
    os.system(f"mkdir {store_dir}/noisy/{timestamp}")
    os.system(f"mkdir {store_dir}/target/{timestamp}")
    os.system(f"mkdir {store_dir}/torch/{timestamp}")
except:
    pass


def background(x, a, b):
    xhalfidx = np.argmin(abs(x-x[len(x)//2]))
    xhalf = x[xhalfidx]
    return a - b*((x - xhalf)/10)**2
    

def lorentzian(x, x0, gamma, amp=1.0):
    return amp*gamma/2/np.pi/((x-x0)**2 + gamma*gamma/4)


def get_realization(sig, max_scale=20., noise_scale=3.):
    max_noise = sig.max()/max_scale
    noisy_sig = sig + np.random.randn(len(sig))*sig/noise_scale
    return noisy_sig


def savenpy(fname, fval):
    np.save(fname, fval)
#    print(f"Writing {fname}")
    return None


x = np.load('data-files/x.npy')
x0_superlist = np.load('data-files/x0.npy')
gamma_superlist = np.load('data-files/gamma.npy')
bg_dcshift_superlist = np.load('data-files/bg_dcshift.npy')
bg_curvature_superlist = np.load('data-files/bg_curvature.npy')
num_lorentzians = params.num_lorentzians
total_samples = 5000 if training_or_testing == 'train' else 500
num_realizations = 50 if training_or_testing == 'train' else 20


synth_datagen_dict = {}
synth_datagen_dict['x'] = x
synth_datagen_dict['x0_superlist'] = x0_superlist
synth_datagen_dict['gamma_superlist'] = gamma_superlist
synth_datagen_dict['bg_dcshift_superlist'] = bg_dcshift_superlist
synth_datagen_dict['bg_curvature_superlist'] = bg_curvature_superlist
synth_datagen_dict['total_samples'] = total_samples
synth_datagen_dict['num_realizations'] = num_realizations
fn.save_obj(synth_datagen_dict, f"{store_dir}/metadata-{timestamp}")




for samplenum in tqdm(range(total_samples), desc='samples'):
    amps = [np.random.uniform(low=0.5, high=1.5) 
            for i in range(num_lorentzians)]
    rlz_super = []
    sig_super = []
    for idx_super, x0list in enumerate(x0_superlist):
        bg_dcshift = bg_dcshift_superlist[idx_super]
        bg_curvature = bg_curvature_superlist[idx_super]
        gammalist = gamma_superlist[idx_super]
        sig = 0.0
        for idx, x0 in enumerate(x0list):
            gamma = gammalist[idx]
            sig += lorentzian(x, x0, gamma, amp=amps[idx]*0.1)
        bg = background(x, bg_dcshift, bg_curvature)*0.1
        sig += bg
    
        rlz_list = []
        for i in range(num_realizations):
            rlz = get_realization(sig)
            rlz_list.append(rlz)
        rlz_super.append(rlz_list)
        sig_super.append(sig - bg)

    rlz_super = np.asarray(rlz_super).astype('float32')
    sig_super = np.asarray(sig_super).astype('float32')

    for i in range(num_realizations):
        datadict = {}
        datadict['noisy'] = np.squeeze(rlz_super[:, i, :])
        datadict['target'] = np.squeeze(sig-bg)
        fname_suffix = f"{samplenum:05d}{i:02d}"
        savenpy(f"{store_dir}/noisy/{timestamp}/synth-{fname_suffix}.npy", np.squeeze(rlz_super[:, i, :]))
        savenpy(f"{store_dir}/target/{timestamp}/true-{fname_suffix}.npy", np.squeeze(sig-bg))
        fn.save_obj(datadict, f"{store_dir}/torch/{timestamp}/data-{fname_suffix}", printop=False)

