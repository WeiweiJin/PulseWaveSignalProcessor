#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 16:37:24 2020

@author: weiweijin
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from scipy.signal import kaiserord, firwin, filtfilt
from scipy import signal

# %% import data
# import waveform data 
PW = pd.read_csv('sample_ppg.csv')

# Plot raw signal
fig = plt.figure(figsize = (18,10))
ax = fig.add_subplot(1,1,1) 
plt.plot(PW.time,PW.ppg, 'k-', linewidth=2)
plt.rc('xtick',labelsize=16)
plt.rc('ytick',labelsize=16)
ax.set_xlabel('time [s]', fontsize = 20)
ax.set_ylabel('PPG [-]', fontsize = 20)
plt.xticks(np.arange(0, 502, 250)) 
plt.yticks(np.arange(-10, 12, 10)) 
filename = 'Raw_signal.png'
fig.savefig(filename)

fig = plt.figure(figsize = (18,10))
ax = fig.add_subplot(1,1,1) 
plt.plot(PW.time[0:250],PW.ppg[0:250], 'k-', linewidth=2)
plt.rc('xtick',labelsize=16)
plt.rc('ytick',labelsize=16)
ax.set_xlabel('time [s]', fontsize = 20)
ax.set_ylabel('PPG [-]', fontsize = 20)
plt.xticks(np.arange(0, 2, 1)) 
plt.yticks(np.arange(-10, 12, 10)) 
filename = 'Raw_signal_sigal.png'
fig.savefig(filename)

# %% Filter out very low and very high frequency noise 
# get ppg data
ppg = PW.loc[:, 'ppg'].values

# Create low pass filter
sample_rate = 300 #sample rat of the signal

nyq_rate = sample_rate / 2.0 # The Nyquist rate of the signal
width = 5.0/nyq_rate #  5 Hz transition width
ripple_db = 8.0 # Attenuation in the stop band
N, beta = kaiserord(ripple_db, width) # Compute the order and Kaiser parameter for the FIR filter.
if N % 2 ==0:
    N = N +1

cutoff_hz_l = 2.0 # The cutoff frequency of the filter
taps = firwin(N, cutoff_hz_l/nyq_rate, window=('kaiser', beta)) # Use firwin with a Kaiser window to create a lowpass FIR filter
# filter out low frequncy siginals
ppg_flt = filtfilt(taps, 1.0, ppg)
# plot filtered signal
fig = plt.figure(figsize = (18,10))
ax = fig.add_subplot(1,1,1) 
plt.plot(PW.time[0:250],ppg_flt[0:250], 'k-', linewidth=2)
plt.rc('xtick',labelsize=16)
plt.rc('ytick',labelsize=16)
ax.set_xlabel('time [s]', fontsize = 20)
ax.set_ylabel('PPG [-]', fontsize = 20)
plt.xticks(np.arange(0, 2, 1)) 
plt.yticks(np.arange(-10, 12, 10)) 
filename = 'LowPass.png'
fig.savefig(filename)

# Create high pass filter

sos = signal.butter(1, 1, 'hp', fs=sample_rate, output='sos')
ppg_flt = signal.sosfilt(sos, ppg_flt)


# plot filtered signal
fig = plt.figure(figsize = (18,10))
ax = fig.add_subplot(1,1,1) 
plt.plot(PW.time,ppg_flt, 'k-', linewidth=2)
plt.rc('xtick',labelsize=16)
plt.rc('ytick',labelsize=16)
ax.set_xlabel('time [s]', fontsize = 20)
ax.set_ylabel('PPG [-]', fontsize = 20)
plt.xticks(np.arange(0, 502, 250)) 
plt.yticks(np.arange(-10, 12, 10)) 
filename = 'HighPass.png'
fig.savefig(filename)

# filter out high frequency again
ppg_flt = filtfilt(taps, 1.0, ppg_flt)
# plot filtered signal
fig = plt.figure(figsize = (18,10))
ax = fig.add_subplot(1,1,1) 
plt.plot(PW.time[0:250],ppg_flt[0:250], 'k-', linewidth=2)
plt.rc('xtick',labelsize=16)
plt.rc('ytick',labelsize=16)
ax.set_xlabel('time [s]', fontsize = 20)
ax.set_ylabel('PPG [-]', fontsize = 20)
plt.xticks(np.arange(0, 2, 1)) 
plt.yticks(np.arange(-10, 12, 10)) 
filename = 'LowPass_final.png'
fig.savefig(filename)


# %% Detect beats
# detect peaks
ppg_avg = np.mean(ppg_flt) #find the mean value of the dataset

grad_ppg = np.gradient(ppg_flt)

grad_ppg_b = grad_ppg[0:-1]
grad_ppg_a = grad_ppg[1:]

grad_sig = np.multiply(grad_ppg_b,grad_ppg_a)

pos_grad = np.argwhere(grad_sig<0) # find peaks and troughs in the waveforms

tmp_peak_id = [] # identify temp peak

for ii in pos_grad:
    if ppg_flt[ii] > ppg_avg:
        tmp_peak_id.extend(ii)

id_dif = np.array(tmp_peak_id[1:]) - np.array(tmp_peak_id[0:-1]) # identify the peaks that are very colse to each other
id_dif = id_dif.reshape(-1)

small_dif = np.argwhere(id_dif < 50)
small_dif = small_dif.reshape(-1)

small_dif_list = small_dif.tolist()

def ranges(nums):
    nums = sorted(set(nums))
    gaps = [[s, e] for s, e in zip(nums, nums[1:]) if s+1 < e]
    edges = iter(nums[:1] + sum(gaps, []) + nums[-1:])
    return list(zip(edges, edges))

cons_range = ranges(small_dif_list)

id_keep = [] # identify the true peak in those close range peaks

for jj in range(len(cons_range)):
    tmp = np.argmax(ppg_flt[(tmp_peak_id[cons_range[jj][0]]):(tmp_peak_id[cons_range[jj][1]+1]+1)])
    tmp = tmp_peak_id[cons_range[jj][0]] + tmp
    id_keep.append(tmp)

def split_list(n):
    """will return the list index"""
    return [(x+1) for x,y in zip(n, n[1:]) if y-x != 1]

def get_sub_list(my_list):
    """will split the list base on the index"""
    my_index = split_list(my_list)
    output = list()
    prev = 0
    for index in my_index:
        new_list = [ x for x in my_list[prev:] if x < index]
        output.append(new_list)
        prev += len(new_list)
    output.append([ x for x in my_list[prev:]])
    return output

cons_list = get_sub_list(small_dif_list)

for nn in range(len(cons_list)):
    cons_list[nn].append(cons_list[nn][-1]+1)

peak_id = tmp_peak_id.copy() # delete all close range peaks

for xx in cons_list:
    for ind in xx:
        peak_id.remove(tmp_peak_id[ind])

peak_id.extend(id_keep) # add back the true peaks

peak_id.sort()

# detect onset
beats = len(peak_id)

onset = [0]

for bt in range(beats-1):
    tmp = np.argmin(ppg_flt[peak_id[bt]:peak_id[bt+1]])
    tmp = peak_id[bt] + tmp
    onset.append(tmp)

# seprate beats
wave = []
for bt in range(beats):
    if bt == beats-1:
        tmp = ppg_flt[onset[bt]:]
    else:
        tmp = ppg_flt[onset[bt]:onset[bt+1]]
    wave.append(tmp)

# plot filtered signal
dt = 1/sample_rate
T = dt * len(wave[122])
time = np.linspace(0,T,num = len(wave[122]))
fig = plt.figure(figsize = (18,10))
ax = fig.add_subplot(1,1,1) 
plt.plot(time,wave[122], 'k-', linewidth=2)
plt.rc('xtick',labelsize=16)
plt.rc('ytick',labelsize=16)
ax.set_xlabel('time [s]', fontsize = 20)
ax.set_ylabel('PPG [-]', fontsize = 20)
plt.xticks(np.arange(0, 2, 1)) 
plt.yticks(np.arange(-10, 12, 10)) 
filename = 'singal_waveform_example.png'
fig.savefig(filename)

