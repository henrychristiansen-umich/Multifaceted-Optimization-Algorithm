"""
MOPT for MOA
----------------------

Author: Henry Christiansen
Date:   2026-07-09
Email:  hennyc@umich.edu

Description:
    This script determines the unbiased ballistic coefficeints for
    all spacecraft of interest for the specified storm. 
    The ballistic coefficients are reported to a file that is then 
    used in FOPT to determine the F10.7 and Ap adjustments.


Usage:
    python mopt.py <Storm> <NRLMSIS Model>
"""

# imports
from scipy.optimize import minimize_scalar
import matplotlib.dates as mdates
from multiprocessing import Pool, cpu_count
from scipy.interpolate import UnivariateSpline
from datetime import timedelta
from collections import defaultdict
from datetime import datetime
import matplotlib.pyplot as plt
from load_gmat import *
from tqdm import tqdm
import numpy as np
import pandas as pd
import json
import os
import re
import shutil
import tempfile
import argparse

def get_tle_data(file, sat_ids, start_date, end_date):
    """
    TLE data processing

    - date filtering
    - ensures chronological sorting
    - duplicate removal
    - minimum 5 points

    Returns:
    dict:
        sat_id: [cleaned TLE entries]
    """

    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    with open(file, 'r') as f:
        tle_data = json.load(f)

    # Only keep requested spacecraft
    sat_ids = set(sat_ids)
    raw_data = defaultdict(list)

    for entry in tle_data:
        sat_id = entry["NORAD_CAT_ID"]
        if sat_id in sat_ids:
            raw_data[sat_id].append(entry)

    filtered_data = {}

    for sat_id in tqdm(raw_data.keys(), desc="Filtering"):

        tle_list = raw_data[sat_id]

        # Filter by date
        pre_filtered = []
        for entry in tle_list:
            epoch = pd.to_datetime(entry["EPOCH"])
            if start_date <= epoch <= end_date:
                pre_filtered.append((epoch, entry))

        # Sort chronologically
        pre_filtered.sort(key=lambda x: x[0])

        # Remove duplicate epochs 
        unique_epochs = set()
        cleaned = []
        for epoch, entry in pre_filtered:
            if epoch not in unique_epochs:
                unique_epochs.add(epoch)
                cleaned.append((epoch, entry))

        # Remove duplicates if not exact epoch
        final = []
        last_time = None

        for epoch, entry in cleaned:
            if last_time is None or (epoch - last_time).total_seconds() >= 5:
                final.append(entry)
                last_time = epoch

        # At least 5 points
        if len(final) < 5:
            continue

        filtered_data[sat_id] = final

    return filtered_data

def find_spacecraft(file1, file2, start_date, mid_date, end_date, sigma, cygnss):
    """
    Reads data, filteres by dsma < 0 and sigma bounds

    """

    start_date = pd.to_datetime(start_date)
    mid_date = pd.to_datetime(mid_date)
    end_date = pd.to_datetime(end_date)

    with open(file1, 'r') as f:
        data1 = json.load(f)

    with open(file2, 'r') as f:
        data2 = json.load(f)

    if cygnss:
        tle_data = data1
    else:
        tle_data = data1 + data2

    # Group by NORAD CATALOG ID
    raw_data = defaultdict(list)
    for entry in tle_data:
        raw_data[entry["NORAD_CAT_ID"]].append(entry)
    
    total_spacecraft = len(raw_data.keys())

    all_times, all_sma = [], []

    phys_times, phys_dsma, phys_dsma_norm, phys_ids = [], [], [], []

    selected_times, selected_dsma = [], []

    picked_spacecraft = []

    for sat_id in tqdm(raw_data.keys(), desc="Filtering"):

        tle_list = raw_data[sat_id]

        # filter by date
        pre_filtered = []
        test_list = []
        test_list2 = []
        for entry in tle_list:
            epoch = pd.to_datetime(entry['EPOCH'])
            if start_date <= epoch <= end_date:
                pre_filtered.append((epoch, entry))
            if start_date <= epoch <= mid_date:
                test_list.append((epoch, entry))
            if mid_date <= epoch <= end_date:
                test_list2.append((epoch, entry))

        if not cygnss and len(test_list) < 5:
            continue

        if not cygnss and len(test_list2) < 5:
            continue

        # Sort by time 
        pre_filtered.sort(key=lambda x: x[0])

        # Remove duplicate data
        unique_epochs = set()
        cleaned = []
        for epoch, entry in pre_filtered:
            if epoch not in unique_epochs:
                unique_epochs.add(epoch)
                cleaned.append((epoch, entry))

        # Remove duplicate data (if epochs dont match exactly)
        final = []
        last_time = None
        for epoch, entry in cleaned:
            if last_time is None or (epoch - last_time).total_seconds() >= 5:
                final.append((epoch, entry))
                last_time = epoch
        
        # at least 5 data points
        if not cygnss and len(final) < 5:
            continue
        
        # extract sma 
        t = []
        s = []
        valid = []

        for epoch, entry in final:
            t.append((epoch - start_date).total_seconds() / 86400.0)
            s.append(float(entry['SEMIMAJOR_AXIS']))
            valid.append(entry)
        
        t = np.array(t)
        s = np.array(s)

        if not cygnss:
            if np.any(s < 6728) or np.any(s > 6828):
                continue
        
        all_times.append(t)
        all_sma.append(s)

        dsma_t, dsma = get_derivative(t, s, cygnss)
        dsma_t, dsma = get_spline(dsma_t, dsma, True)
        dsma = dsma * 1000

        if np.any(dsma > 0):
            continue
            
        dif = dsma.max() - dsma.min()
        
        dsma_norm = (dsma - dsma.min()) / dif

        phys_times.append(dsma_t)
        phys_dsma.append(dsma)
        phys_dsma_norm.append(dsma_norm)
        phys_ids.append((sat_id, valid))
    
    print(f"{len(all_times)}/{total_spacecraft} after initial processing")
    print(f"{len(phys_times)} pass dSMA < 0 filter")

    if len(phys_times) == 0:
        return []

    global_t_min = min(t_arr[0] for t_arr in phys_times)   
    global_t_max = max(t_arr[-1] for t_arr in phys_times)  
    common_t = np.arange(global_t_min, global_t_max, 0.01)
    interp_dsma = []

    for t_arr, d_arr in zip(phys_times, phys_dsma_norm):
        try:
            interp = np.interp(common_t, t_arr, d_arr)
            interp_dsma.append(interp)
        except:
            continue

    interp_dsma = np.array(interp_dsma)

    if interp_dsma.shape[0] == 0:
        return []

    medians = np.median(interp_dsma, axis=0)
    stds = np.std(interp_dsma, axis=0)

    upper_bound = medians + sigma * stds
    lower_bound = medians - sigma * stds

    for (sat_id, _), t_arr, d_arr in zip(phys_ids, phys_times, phys_dsma_norm):

        interp = np.interp(common_t, t_arr, d_arr)
        outside = (interp < lower_bound) | (interp > upper_bound)

        if not np.any(outside):
            picked_spacecraft.append(sat_id)
            selected_times.append(t_arr)
            selected_dsma.append(d_arr)

    print(f"{len(selected_times)} pass sigma filter")

    # convert t in days to python datetime
    def to_datetime(t_array):
        return [start_date + pd.Timedelta(days = float(t)) for t in t_array]
    
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True)
    axes = axes.flatten()
    size = 14

    # 1. SMA (all)
    for t, s in zip(all_times, all_sma):
        if cygnss:
            axes[0].plot(to_datetime(t), s, 'k-', alpha=1)
        else:
            axes[0].plot(to_datetime(t), s, 'k-', alpha=0.05)
    axes[0].set_ylabel("SMA (km)", fontweight='bold', fontsize = size)
    axes[0].set_title(f"(a) TLE data for all {len(all_sma)} spacecraft", fontweight='bold', fontsize=size)

    # 2. dSMA (all)
    for t, d in zip(phys_times, phys_dsma):
        if cygnss:
            axes[1].plot(to_datetime(t), d, 'k-', alpha=1)
        else:
            axes[1].plot(to_datetime(t), d / 1000, 'k-', alpha=0.1)
    axes[1].set_ylabel("dSMA (km/day)", fontweight='bold', fontsize=size)
    axes[1].set_title("(b) Spacecraft with dSMA < 0", fontweight='bold', fontsize=size)

    # 3. Normalized dSMA (all)
    for t, d in zip(phys_times, phys_dsma_norm):
        if cygnss:
            axes[2].plot(to_datetime(t), d, 'k-', alpha=1)
        else:
            axes[2].plot(to_datetime(t), d, 'k-', alpha=0.1)

    bin_times = to_datetime(common_t)
    axes[2].plot(bin_times, upper_bound, 'r-', linewidth=1, label='+ 1σ')
    axes[2].plot(bin_times, lower_bound, 'r-', linewidth=1, label='- 1σ')

    axes[2].set_ylabel("Normalized dSMA", fontweight='bold', fontsize=size)
    axes[2].set_title(f"(c) Spacecraft within {sigma}σ of median", fontweight='bold', fontsize=size)
    # axes[2].legend(fontsize=size)

    # 4. Normalized dSMA (filtered only)
    for t, d in zip(selected_times, selected_dsma):
        if cygnss:
            axes[3].plot(to_datetime(t), d, 'k-', alpha = 1)
        else:
            axes[3].plot(to_datetime(t), d, 'k-', alpha = 0.75)

    axes[3].set_ylabel("Normalized dSMA", fontweight='bold', fontsize=size)
    axes[3].set_title(f"(d) Selected {len(selected_dsma)} spacecraft", fontweight='bold', fontsize=size)
    axes[2].set_xlabel("Real Time", fontweight='bold', fontsize=size)
    axes[3].set_xlabel("Real Time", fontweight='bold', fontsize=size)

    for ax in axes:
        ax.tick_params(axis='both', labelsize=size)
        ax.set_xlim(start_date, end_date)
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=4))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    
    plt.tight_layout()

    if cygnss:
        plt.savefig(f"{BASE_PATH}/Results/mopt_cygnss.png")
    else:
        plt.savefig(f"{BASE_PATH}/Results/filter.png", dpi=300)

    # plt.show()
    plt.close(fig)

    return picked_spacecraft
    
def find_mass(args):
    """

    """
    # determine tle_dsma and tle_time
    tle_data, sat_id, start_date, end_date, tol, max_iter, gmat_template, cygnss = args

    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    tle_sma_t = []
    tle_sma = []

    for entry in tle_data:
        epoch = pd.to_datetime(entry['EPOCH'])
        tle_sma_t.append((epoch - start_date).total_seconds() / 86400.0)
        tle_sma.append(float(entry["SEMIMAJOR_AXIS"]))
        if not start_date <= epoch <= end_date:
            print("TLE DATA out of range - shouldn't happen")
            sys.exit()
    
    tle_sma_t = np.array(tle_sma_t)
    tle_sma = np.array(tle_sma)

    tle_dsma_t, tle_dsma = get_derivative(tle_sma_t, tle_sma, cygnss)
    tle_dsma_t, tle_dsma = get_spline(tle_dsma_t, tle_dsma, False)
    tle_dsma = tle_dsma * 1000

    # Prepare TEMP GMAT SCRIPT and Directory
    base_temp_folder = "/home/hennyc/temp"
    os.makedirs(base_temp_folder, exist_ok=True)

    temp_dir = tempfile.mkdtemp(prefix=f'gmat_{sat_id}_', dir=base_temp_folder)
    script_path = os.path.join(temp_dir, f'gmat_sat_{sat_id}.script')
    output_path = os.path.join(temp_dir, f'output_{sat_id}.txt')
    with open(gmat_template, 'r') as f:
        script = f.read()
    
    # Prepare GMAT Script for propogation
    epoch = datetime.strptime(tle_data[0]['EPOCH'], "%Y-%m-%dT%H:%M:%S.%f")
    epoch_string = epoch.strftime("%d %b %Y %H:%M:%S.") + f"{epoch.microsecond // 1000:03d}"
    values = {
        'EPOCH_VAL': epoch_string,
        'SMA_VAL': float(tle_data[0]['SEMIMAJOR_AXIS']),
        'ECC_VAL': float(tle_data[0]['ECCENTRICITY']),
        'INC_VAL': float(tle_data[0]['INCLINATION']),
        'RAAN_VAL': float(tle_data[0]['RA_OF_ASC_NODE']),
        'AOP_VAL': float(tle_data[0]['ARG_OF_PERICENTER']),
        'MA_VAL': float(tle_data[0]['MEAN_ANOMALY']),
        'FILENAME_VAL': output_path,
        'LENGTH_VAL': round(((np.datetime64(tle_data[-1]['EPOCH']) - np.datetime64
                              (tle_data[0]['EPOCH'])) / np.timedelta64(1, 'D')),8)
    }

    for key, val in values.items():
        script = script.replace(key, str(val))
        
    with open(script_path, 'w') as f:
        f.write(script)

    # Bounded optimization to minimize the RMS error by changing the mass
    result = minimize_scalar(
        lambda mass: rms_error(mass, script_path, output_path, tle_sma_t, tle_sma, tle_dsma_t, tle_dsma),
        bounds=(0.1, 10000),
        method='bounded',
        options={'xatol': tol, 'maxiter': max_iter}
    )

    shutil.rmtree(temp_dir)
    if result.success:
        return result.x
    else:
        return None

def get_derivative(x_arr, y_arr, cygnss):
    x_mid = (x_arr[:-1] + x_arr[1:]) / 2
    dsma = (y_arr[1:] - y_arr[:-1]) / (x_arr[1:] - x_arr[:-1])
    if cygnss:        
        mask = dsma <= 0 
        x_mid = x_mid[mask]
        dsma = dsma[mask]
    return np.array(x_mid), np.array(dsma)

def rms_error(mass, script_path, output_path, tle_sma_t, tle_sma, tle_dsma_t, tle_dsma):
    """

    """
    try:
        gmat.LoadScript(script_path)
        sat = gmat.GetObject("Sat")
        sat.SetField("DryMass", mass)
        gmat.Initialize()
        status = gmat.Execute()
        if status != 1:
            return np.nan
    except Exception as _:
        return np.nan
    finally:
        gmat.Clear()
    
    # Read Data
    output_data = np.loadtxt(output_path, delimiter=',')
    time = np.array(output_data[:, 0])
    time = time + tle_sma_t[0]
    sma = np.array(output_data[:,1])

    ma = np.mod(output_data[:, 2], 360)

    wraps = np.diff(ma) < -300 
    orbit_id = np.cumsum(np.insert(wraps, 0, False))
    counts = np.bincount(orbit_id)
    sma_orbit_avg = np.bincount(orbit_id, weights=sma) / counts
    time_orbit_avg = np.bincount(orbit_id, weights=time) / counts

    # combine every k orbits
    k = 5
    n = len(sma_orbit_avg) // k  

    sma_orbit_avg = sma_orbit_avg[:n*k].reshape(n, k).mean(axis=1)
    time_orbit_avg = time_orbit_avg[:n*k].reshape(n, k).mean(axis=1)
    

    gmat_spline = UnivariateSpline(time_orbit_avg, sma_orbit_avg, s = 0)
    gmat_dsma = gmat_spline.derivative()(tle_dsma_t) * 1000
    gmat_sma = gmat_spline(tle_sma_t)
    gmat_sma = np.array(gmat_sma)

    dif = tle_sma[0] - gmat_sma[0]
    gmat_sma = gmat_sma + dif

    return np.sqrt(np.mean((tle_dsma - gmat_dsma) ** 2))

def get_spline(x, y, fopt):
    y_min = np.min(y)
    y_max = np.max(y)
    y_norm = (y - y_min) / (y_max - y_min)

    median = np.median(y_norm)
    mad = np.median(np.abs(y_norm - median))  # robust scale
    sigma_robust = 1.4826 * mad          # convert MAD → sigma

    mean = np.mean(y_norm)
    w = np.ones_like(y_norm) * (1.0 / sigma_robust)
    if fopt:
        neg_outliers = y_norm < (mean - 2 * sigma_robust)
        w[neg_outliers] *= 10.0   # tune factor as needed # was 100
    spline = UnivariateSpline(x, y_norm, w=w, s=len(x))
    spline_zero = UnivariateSpline(x, y_norm, s = 0)
    
    x_ret = np.arange(x[0], x[-1], 0.01)
    y_ret1 = spline(x_ret)
    y_ret2 = spline_zero(x_ret)

    y_ret = (0.5*y_ret1 + 0.5*y_ret2)

    return x_ret, y_ret2 * (y_max - y_min) + y_min

def weight_plot(data, start_date, storm):
    ids = list(data.keys())

    for sat_id in ids:
        tle_data = data[sat_id]

        tle_sma_t = []
        tle_sma = []

        for entry in tle_data:
            epoch = pd.to_datetime(entry['EPOCH'])
            tle_sma_t.append((epoch - start_date).total_seconds() / 86400.0)
            tle_sma.append(float(entry["SEMIMAJOR_AXIS"]))

        tle_sma_t = np.array(tle_sma_t)
        tle_sma = np.array(tle_sma)

        tle_dsma_t, tle_dsma = get_derivative(tle_sma_t, tle_sma, False)

        spl = UnivariateSpline(tle_dsma_t, tle_dsma, s=0)
        tle_dsma_t, tle_dsma = get_spline(tle_dsma_t, tle_dsma, True)
        tle_dsma = tle_dsma * 1000
        tle_dsma2 = spl(tle_dsma_t) * 1000

        if np.any(tle_dsma > 0):
            continue

        plt.plot(tle_dsma_t, tle_dsma, alpha=1)
        plt.plot(tle_dsma_t, tle_dsma2, 'k--', alpha = 0.3)

    plt.xlabel("Time")
    plt.ylabel("dSMA")
    plt.title("All Selected Spacecraft FOPT/AOPT Splines")
    plt.savefig(f"{BASE_PATH}/Results/splines.png")
    # plt.show()

if __name__ == '__main__':

    """
    Main execution logic for MOPT.

    """

    rel_path = "./" # Path to data Folder

    parser = argparse.ArgumentParser()
    parser.add_argument("date")
    parser.add_argument("model")
    args = parser.parse_args()
    storm_name = args.date

    STORM_FIGURE_STR = storm_name
    atmospheric_model = args.model

    if atmospheric_model not in ("MSISE90", "NRLMSISE00"):
        print("Error: unknown <NRLMSISE Model>. Use MSISE90 or NRLMSISE00")
        print("Usage: python mopt.py <name> <NRLMSISE Model>")
        sys.exit(1)

    # CYGNSS spacecraft ballistic coefficient
    cygnss_bc = 0.0130

    # Spacecraft filtering (reduces junk TLE data)
    filter_sigma = 1.5

    # Maximum # of iterations for mass determination and mass tolerance
    max_iterations = 50
    mass_tolerance = 0.1 # kg

    # define relative file paths
    BASE_PATH = f'{rel_path}/{storm_name}'

    date_path = f'{BASE_PATH}/DATES.txt'
    tle_path1 = f'{BASE_PATH}/TLE_DATA_MOPT.json'
    tle_path2 = f'{BASE_PATH}/TLE_DATA_FOPT.json'
    output_path = f'{BASE_PATH}/MOPT_OUTPUT.txt'
    gmat_script_path = '/home/hennyc/src/mopt.script'
    ref_path = f'{BASE_PATH}/CYGNSS.json'

    # Modify mopt.script with inputted atmospheric model
    with open(gmat_script_path, 'r') as f:
        script = f.read()
    script = re.sub(r"\b(MSISE90|NRLMSISE00|MSIS21)\b", "ATM_MOD_VAL", script)
    script = script.replace("ATM_MOD_VAL", str(atmospheric_model))
    with open(gmat_script_path, 'w') as f:
        f.write(script)

    # Determine MOPT start and end date from date file
    with open(date_path, "r") as file:
        _, start_str_mopt, end_str_mopt = file.readline().strip().split(",")
        _, start_str_fopt, end_str_fopt = file.readline().strip().split(",")
    start_date_mopt, end_date_mopt, start_date_fopt, end_date_fopt = np.datetime64(start_str_mopt), np.datetime64(end_str_mopt), np.datetime64(start_str_fopt), np.datetime64(end_str_fopt)

    # Read CYGNSS TLE Data
    # ref_ids = find_spacecraft(ref_path, ref_path, start_date_mopt, end_date_mopt, end_date_fopt, 100, True)
    ref_ids = ['41884','41885','41886','41887','41888','41889','41890','41891']
    ref_data = get_tle_data(ref_path, ref_ids, start_date_mopt, end_date_mopt)

    # Determine ballistic coefficeint for CYGNSS Spacecraft
    ref_args = [(ref_data[ref_id], ref_id, start_date_mopt, end_date_mopt, mass_tolerance, max_iterations, gmat_script_path, True) for ref_id in ref_ids]
    
    with Pool(processes=cpu_count()) as pool:
        ref_masses = list(tqdm(pool.imap(find_mass, ref_args), total=len(ref_args), desc="CYGNSS"))
    
    ref_masses = [r for r in ref_masses if r is not None]

    # Determine mass (ballistic coefficient) adjustment
    ref_masses = np.array(ref_masses)
    ref_masses = np.round(ref_masses,1)
    vals = 1 / ref_masses
    ref_median = np.round(np.median(vals),5)
    mass_adjustment = ref_median/cygnss_bc
    print(ref_masses)
    print(mass_adjustment)

    # Read TLE data for spacecraft
    ids = find_spacecraft(tle_path1, tle_path2, start_date_mopt, end_date_mopt, end_date_fopt, filter_sigma, False)
    # if len(ids) < 30:
    #     ids = find_spacecraft(tle_path1, tle_path2, start_date_mopt, end_date_mopt, end_date_fopt, filter_sigma + 0.25, False)
    # if len(ids) < 30:
    #     ids = find_spacecraft(tle_path1, tle_path2, start_date_mopt, end_date_mopt, end_date_fopt, filter_sigma + 0.5, False)

    data = get_tle_data(tle_path1, ids, start_date_mopt, end_date_mopt)

    data_for_weight_plot = get_tle_data(tle_path2, ids, start_date_fopt, end_date_fopt)
    weight_plot(data_for_weight_plot, start_date_fopt, storm_name)

    # Determine ballistic coefficeints for spacecraft
    ids = list(data.keys())
    args_list = [(data[sat_id], sat_id, start_date_mopt, end_date_mopt, mass_tolerance, max_iterations, gmat_script_path, False) for sat_id in ids]
    
    with Pool(processes=cpu_count()) as pool:
        mass = list(tqdm(pool.imap(find_mass, args_list), total=len(args_list), desc="MOPT"))

    masses = [r for r in mass if r is not None]

    masses = np.array(masses)*mass_adjustment

    print(f"{len(masses)}/{len(mass)} Converged")

    with open(output_path, 'w') as file:
        for id, m in zip(ids, masses):
            file.write(f"{id},{m}\n")
