# MOPT Solver for MOA 2
#
# Coded by Henry Christiansen. University of Michigan.
#
# This file solves for the mass of spacecraft given TLE data and 
# assuming Cd = 2.2 and A = 1 m^2

#imports
from matplotlib.ticker import MultipleLocator
from scipy.optimize import minimize_scalar
from multiprocessing import Pool, cpu_count
from scipy.ndimage import gaussian_filter1d
from scipy.stats import linregress
from collections import defaultdict
from datetime import datetime
from load_gmat import *
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
import os
import shutil
import tempfile

def read_tle_file(storm, file, start_date, end_date, sigma):
    with open(file, 'r') as f:
        tle_data = json.load(f)

    raw_data = defaultdict(list)
    for entry in tle_data:
        name = entry["NORAD_CAT_ID"]
        raw_data[name].append(entry)

    filtered_data = defaultdict(list)
    sma_times = []
    sma_values = []
    ids = []

    for id in tqdm(raw_data.keys(), desc="Filtering"):
        tle_list = raw_data[id]
        epochs = []
        tle_sma = []
        tle_time = []
        valid = []

        for i in tle_list:
            dt = pd.to_datetime(i['EPOCH'])
            too_close = any(abs((dt - existing).total_seconds()) < 0.5 * 3600 for existing in epochs)
            if too_close:
                continue
            epochs.append(dt)

            epoch = np.datetime64(i['EPOCH'])
            if start_date <= epoch <= end_date:
                tle_sma.append(float(i['SEMIMAJOR_AXIS']))
                tle_time.append((epoch - start_date) / np.timedelta64(1, 'D'))
                valid.append(i)

        if len(tle_sma) < int(((end_date - start_date) / np.timedelta64(1, 'D'))/8):
            continue

        sma_times.append(np.array(tle_time))
        sma_values.append(np.array(tle_sma))
        ids.append((id, valid))

    filtered_sma_times = []
    filtered_sma_values = []
    filtered_dsma_values = []

    for t, s in zip(sma_times, sma_values):
        dsma = np.gradient(s, t)
        min_val = dsma.min()
        if min_val == 0:
            continue  

        dsma_norm = dsma / abs(min_val)

        if np.any(dsma_norm > 0):
            continue  

        filtered_sma_times.append(t)
        filtered_sma_values.append(s)
        filtered_dsma_values.append(dsma_norm)

    sma_times = filtered_sma_times
    sma_values = filtered_sma_values
    dsma_values = filtered_dsma_values

    all_times = np.concatenate(sma_times)
    t_min = np.floor(all_times.min())
    t_max = np.ceil(all_times.max())
    bin_edges = np.arange(t_min, t_max + 0.125, 0.125)

    bin_stats = []
    for left, right in zip(bin_edges[:-1], bin_edges[1:]):
        bin_vals = []
        for times_arr, dsma_arr in zip(sma_times, dsma_values):
            mask = (times_arr >= left) & (times_arr < right)
            bin_vals.extend(dsma_arr[mask])
        if bin_vals:
            median = np.median(bin_vals)
            std = np.std(bin_vals)
            bin_stats.append((median, std))
        else:
            bin_stats.append((np.nan, np.nan))

    for (id, valid_tle), times_arr, dsma_arr in zip(ids, sma_times, dsma_values):
        outlier = False
        for t, val in zip(times_arr, dsma_arr):
            bin_idx = np.searchsorted(bin_edges, t, side='right') - 1
            if 0 <= bin_idx < len(bin_stats):
                median, std = bin_stats[bin_idx]
                if not np.isnan(median) and std > 0:
                    lower = median - sigma * std
                    upper = median + sigma * std
                    if not (lower <= val <= upper):
                        outlier = True
                        break
        if not outlier:
            filtered_data[id].extend(valid_tle)
    
    plt.figure(figsize=(11, 7))

    for times_arr, dsma_arr in zip(sma_times, dsma_values):
        if len(times_arr) > 0:
            plt.plot(times_arr, dsma_arr, marker='o', color='black', alpha=0.3, linewidth=1)

    for (id, valid_tle), times_arr, dsma_arr in zip(ids, sma_times, dsma_values):
        if id in filtered_data:
            if len(times_arr) > 0:
                plt.plot(times_arr, dsma_arr, marker='o',color='black', alpha=1.0, linewidth=5.0)

    bin_medians = [b[0] for b in bin_stats]
    bin_dates = bin_edges[:-1] 
    plt.step(bin_dates, bin_medians, where='post', color='red', linewidth=3.0, label='Median (3-hr post)')

    plt.gca().xaxis.set_major_locator(MultipleLocator(1))
    plt.xlabel(f"Days Since {start_date}")
    plt.ylabel("Normalized dSMA")
    plt.title(f"Normalized dSMA of TLEs, Sigma ={sigma} ({len(filtered_data)} sats)")
    plt.ylim([0, -1])
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'../STORMS/{storm}/Results/mopt.png')

    return filtered_data
    
def find_mass(args):
    # Sorting TLE_DATA
    tle_data, sat_id, start_date, end_date, tol, max_iter, gmat_template = args
    epochs = np.array([x['EPOCH'] for x in tle_data], dtype='datetime64')
    ind = np.argsort(epochs)
    sorted_tle_data = [tle_data[i] for i in ind]

    # Find TLE dSMA
    tle_sma, tle_time, filtered_sorted_tle_data = [], [], []
    for i in sorted_tle_data:
        epoch = np.datetime64(i['EPOCH'])
        if (start_date <= epoch <= end_date):
            tle_sma.append(float(i['SEMIMAJOR_AXIS']))
            filtered_sorted_tle_data.append(i)
    
    if len(filtered_sorted_tle_data) < 5:
        return False
    
    start = np.datetime64(filtered_sorted_tle_data[0]['EPOCH'])
    for i in filtered_sorted_tle_data:
        epoch = np.datetime64(i['EPOCH'])
        tle_time.append((epoch - start) / np.timedelta64(1, 'D'))
    
    tle_dsma = np.gradient(tle_sma, tle_time) * 365 #km/year

    # Prepare TEMP GMAT SCRIPT and Directory
    temp_dir = tempfile.mkdtemp(prefix=f'gmat_{sat_id}_')
    script_path = os.path.join(temp_dir, f'gmat_sat_{sat_id}.script')
    output_path = os.path.join(temp_dir, f'output_{sat_id}.txt')
    with open(gmat_template, 'r') as f:
        script = f.read()
        
    epoch = datetime.strptime(filtered_sorted_tle_data[0]['EPOCH'], "%Y-%m-%dT%H:%M:%S.%f")
    epoch_string = epoch.strftime("%d %b %Y %H:%M:%S.") + f"{epoch.microsecond // 1000:03d}"
    values = {
        'EPOCH_VAL': epoch_string,
        'SMA_VAL': float(filtered_sorted_tle_data[0]['SEMIMAJOR_AXIS']),
        'ECC_VAL': float(filtered_sorted_tle_data[0]['ECCENTRICITY']),
        'INC_VAL': float(filtered_sorted_tle_data[0]['INCLINATION']),
        'RAAN_VAL': float(filtered_sorted_tle_data[0]['RA_OF_ASC_NODE']),
        'AOP_VAL': float(filtered_sorted_tle_data[0]['ARG_OF_PERICENTER']),
        'MA_VAL': float(filtered_sorted_tle_data[0]['MEAN_ANOMALY']),
        'FILENAME_VAL': output_path,
        'LENGTH_VAL': round(((np.datetime64(filtered_sorted_tle_data[-1]['EPOCH']) - np.datetime64
                              (filtered_sorted_tle_data[0]['EPOCH'])) / np.timedelta64(1, 'D')),8)
    }

    for key, val in values.items():
        script = script.replace(key, str(val))
        
    with open(script_path, 'w') as f:
        f.write(script)

    result = minimize_scalar(
        lambda mass: rms_error(mass, script_path, output_path, tle_time, tle_dsma),
        bounds=(0.1, 1000),
        method='bounded',
        options={'xatol': tol, 'maxiter': max_iter}
    )

    shutil.rmtree(temp_dir)
    if result.success:
        return result.x
    else:
        return False

def rms_error(mass, script_path, output_path, tle_time, tle_dsma):
    try:
        gmat.LoadScript(script_path)
        sat = gmat.GetObject("Sat")
        sat.SetField("DryMass", mass)
        gmat.Initialize()
        status = gmat.Execute()
        if status != 1:
            return np.nan
    except Exception as e:
        return np.nan
    finally:
        gmat.Clear()
    
    # Read Data
    output_data = np.loadtxt(output_path, delimiter=',')
    time = output_data[:, 0]
    sma = output_data[:,1]
    gmat_sma_avg = gaussian_filter1d(sma, sigma=120)
    gmat_dsma = []
    for i, t in enumerate(tle_time):
        t_min, t_max = find_window(i, t, tle_time)
        mask = (time > t_min) & (time < t_max)
        if np.sum(mask) >= 2:
            x = time[mask]
            y = gmat_sma_avg[mask]
            slope, *_ = linregress(x, y)
            gmat_dsma.append(slope)
        else:
            return np.nan
    gmat_dsma = np.array(gmat_dsma) * 365

    return np.sqrt(np.mean((tle_dsma - gmat_dsma) ** 2))

def find_window(i, t, tle_time):
    min = 0
    max = 0
    if i == 0:
        right_window = tle_time[2] - tle_time[0]
        min = t
        max = t + right_window
    elif i == len(tle_time)-1:
        left_window = tle_time[-1] - tle_time[-3]
        min = t - left_window
        max = t
    else:
        left_window = t - tle_time[i-1]
        right_window = tle_time[i+1] - t
        min = t - left_window
        max = t + right_window

    return min, max

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python mopt.py <month_year>")
        sys.exit(1)

    cygnss_bc = 0.013
    filter_sigma = 1.5
    max_iterations = 30
    mass_tolerance = 0.001 # kg
    
    storm_name = sys.argv[1]
    base_path = f'../STORMS/{storm_name}'
    date_path = f'{base_path}/DATES.txt'
    tle_path = f"{base_path}/TLE_DATA.json"
    output_path = f"{base_path}/MOPT_OUTPUT.txt"
    gmat_script_path = 'mopt.script'
    ref_path = "../STORMS/CYGNSS.json"

    with open(date_path, "r") as file:
        _, start_str_1, end_str_1 = file.readline().strip().split(",")
        _, _, end_str_2 = file.readline().strip().split(",")
    start_date, end_date = np.datetime64(start_str_1), np.datetime64(end_str_1)

    ref_data = read_tle_file(storm_name,ref_path, start_date, np.datetime64(end_str_2), 100)
    ref_ids = list(ref_data.keys())
    ref_args = [(ref_data[ref_id], ref_id, start_date, end_date, mass_tolerance, max_iterations, gmat_script_path) for ref_id in ref_ids]
    with Pool(cpu_count()) as pool:
        ref_masses = list(tqdm(pool.imap(find_mass, ref_args), total=len(ref_args), desc="CYGNSS"))
    
    ref_masses = np.array(ref_masses)
    ref_masses = np.round(ref_masses,3)
    ref_median = np.round(np.median(2.2/ref_masses),3)
    mass_adjustment = ref_median/cygnss_bc

    data = read_tle_file(storm_name, tle_path, start_date, np.datetime64(end_str_2), filter_sigma)
    ids = list(data.keys())
    args_list = [(data[sat_id], sat_id, start_date, end_date, mass_tolerance, max_iterations, gmat_script_path) for sat_id in ids]
    with Pool(cpu_count()) as pool:
        masses = list(tqdm(pool.imap(find_mass, args_list), total=len(args_list), desc="MOPT"))

    masses = np.array(masses)*mass_adjustment

    with open(output_path, 'w') as file:
        for id, mass in zip(ids, masses):
            file.write(f"{id},{'NO CONVERGENCE' if not mass else mass}\n")
