"""
MOPT for MOA-2
----------------------

Author: Henry Christiansen
Date:   2026-01-15
Email:  hennyc@umich.edu

Description:
    This script determines the unbiased ballistic coefficeints for
    all spacecraft of interest for the specified storm (<month-year>). 
    The ballistic coefficients are reported to a file that is then 
    used in FOPT to determine the F10.7 and Ap adjustments.


Usage:
    python mopt.py <Storm> <NRLMSIS Model>
"""

# imports
from scipy.optimize import minimize_scalar
from multiprocessing import Pool, cpu_count
from scipy.ndimage import gaussian_filter1d
from scipy.stats import linregress
from collections import defaultdict
from datetime import datetime
from load_gmat import *
from tqdm import tqdm
import numpy as np
import pandas as pd
import json
import os
import re
import shutil
import tempfile

def read_tle_file(storm, file, start_date, end_date, sigma):
    """
    Docstring for read_tle_file: Reads, filters TLE data from .json file.
    
    :param storm: identifier for the current storm (used for plotting data if needed)
    :param file: path to the TLE JSON file
    :param start_date: start date for MOPT analysis
    :param end_date: end date for MOPT analysis
    :param sigma: Threshold for standard deviation to identify TLE outliers. 
                  These outliers are removed.
    """

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

    return filtered_data
    
def find_mass(args):
    """
    Determines the mass (essentially the ballistic coefficeint) for given satellite.

    Parameters
    ----------
    args : tuple
        Tuple containing the following elements:
        - tle_data : list of dict
            TLE data for the satellite. Each dict should have keys such as 'EPOCH',
            'SEMIMAJOR_AXIS', 'ECCENTRICITY', 'INCLINATION', 'RA_OF_ASC_NODE',
            'ARG_OF_PERICENTER', and 'MEAN_ANOMALY'.
        - sat_id : str or int
            Unique identifier for the satellite.
        - start_date : datetime64
            Start of the analysis period.
        - end_date : datetime64
            End of the analysis period.
        - tol : float
            Tolerance for the optimizer.
        - max_iter : int
            Maximum number of iterations for the optimizer.
        - gmat_template : str
            Path to the GMAT template script to use for propogation.

    Returns
    -------
    float or bool
        - Estimated satellite dry mass (float) if the optimization succeeds.
        - False if there is insufficient TLE data (<5 points) or if the optimizer fails.

    Notes
    -----
    - Delta-SMA (dSMA) is computed in km/year.
    - Temporary GMAT scripts and directories are automatically removed after optimization.
    - The function uses 'minimize_scalar' from SciPy to perform bounded optimization.
    """

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
    base_temp_folder = "/home/hennyc/temp"
    os.makedirs(base_temp_folder, exist_ok=True)

    temp_dir = tempfile.mkdtemp(prefix=f'gmat_{sat_id}_', dir=base_temp_folder)
    script_path = os.path.join(temp_dir, f'gmat_sat_{sat_id}.script')
    output_path = os.path.join(temp_dir, f'output_{sat_id}.txt')
    with open(gmat_template, 'r') as f:
        script = f.read()
    
    # Prepare GMAT Script for propogation
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

    # Bounded optimization to minimize the RMS error by changing the mass
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
    """
    Determine the RMS error between the modeled trajectory and the TLE

    Parameters
    ----------
    mass : float
        Current Dry mass of the satellite to use in the GMAT simulation.
    script_path : str
        Path to the GMAT script file to execute.
    output_path : str
        Path to the CSV file generated by GMAT containing time and SMA data.
    tle_time : array
        Sequence of TLE times corresponding to the satellite data.
    tle_dsma : array
        Sequence of dSMA values derived from TLEs for comparison.

    Returns
    -------
    float
        RMS error between modeled dSMA and TLE dSMA. Returns np.nan if
        the simulation fails or insufficient data is available for slope calculation.

    Notes
    -----
    - Uses Gaussian smoothing on SMA data to reduce noise.
    - Slope calculation is done via linear regression over each TLE-defined window.

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
    """
    Determine the time window between two consecutive TLEs

    Parameters:
    -----------
    i : int
        Index of the current TLE in the tle_time list.
    t : float
        The time of the current TLE.
    tle_time : list of float
        Ordered list of TLE times

    Returns:
    --------
    min : float
        Start of the time window for this TLE.
    max : float
        End of the time window for this TLE.

    Notes:
    ------
    - If the current TLE is the first entry, the window extends forward only.
    - If the current TLE is the last entry, the window extends backward only.
    - Otherwise, the window extends symmetrically based on neighboring TLEs.
    """

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

    """
    Main execution logic for MOPT.

    """

    if len(sys.argv) != 3:
        print("Usage: python mopt.py <month_year> <NRLMSISE Model>")
        sys.exit(1)

    storm_name = sys.argv[1]
    atmospheric_model = sys.argv[2]

    if atmospheric_model not in ("MSISE90", "MSIS21"):
        print("Error: unknown <NRLMSISE Model>. Use MSISE90 or MSIS21")
        print("Usage: python mopt.py <month_year> <NRLMSISE Model>")
        sys.exit(1)

    # CYGNSS spacecraft ballistic coefficient
    cygnss_bc = 0.013

    # Spacecraft filtering (reduces junk TLE data)
    filter_sigma = 2

    # Maximum # of iterations for mass determination and mass tolerance
    max_iterations = 30
    mass_tolerance = 1 # kg

    # define relative file paths
    base_path = f'/home/hennyc/data/{storm_name}'
    date_path = f'{base_path}/DATES.txt'
    tle_path = f'{base_path}/TLE_DATA.json'
    output_path = f'{base_path}/MOPT_OUTPUT.txt'
    gmat_script_path = '/home/hennyc/src/mopt.script'
    ref_path = "/home/hennyc/data/CYGNSS.json"

    # Modify mopt.script with inputted atmospheric model
    with open(gmat_script_path, 'r') as f:
        script = f.read()
    script = re.sub(r"\b(MSISE90|MSIS21)\b", "ATM_MOD_VAL", script)
    script = script.replace("ATM_MOD_VAL", str(atmospheric_model))
    with open(gmat_script_path, 'w') as f:
        f.write(script)

    # Determine MOPT start and end date from date file
    with open(date_path, "r") as file:
        _, start_str_1, end_str_1 = file.readline().strip().split(",")
        _, _, end_str_2 = file.readline().strip().split(",")
    start_date, end_date = np.datetime64(start_str_1), np.datetime64(end_str_1)

    # Read CYGNSS TLE Data
    ref_data = read_tle_file(storm_name,ref_path, start_date, np.datetime64(end_str_2), 100)

    # Determine ballistic coefficeint for CYGNSS Spacecraft
    ref_ids = list(ref_data.keys())
    ref_args = [(ref_data[ref_id], ref_id, start_date, end_date, mass_tolerance, max_iterations, gmat_script_path) for ref_id in ref_ids]
    with Pool(processes=cpu_count()) as pool:
        ref_masses = list(tqdm(pool.imap(find_mass, ref_args), total=len(ref_args), desc="CYGNSS"))
    
    # Determine mass (ballistic coefficient) adjustment
    ref_masses = np.array(ref_masses)
    ref_masses = np.round(ref_masses,3)
    ref_median = np.round(np.median(2.2/ref_masses),3)
    mass_adjustment = ref_median/cygnss_bc

    # Read TLE data for spacecraft
    data = read_tle_file(storm_name, tle_path, start_date, np.datetime64(end_str_2), filter_sigma)

    # Determine ballistic coefficeints for spacecraft
    ids = list(data.keys())
    args_list = [(data[sat_id], sat_id, start_date, end_date, mass_tolerance, max_iterations, gmat_script_path) for sat_id in ids]
    with Pool(processes=cpu_count()) as pool:
        masses = list(tqdm(pool.imap(find_mass, args_list), total=len(args_list), desc="MOPT"))

    # Apply mass adjustment
    masses = np.array(masses)*mass_adjustment

    # Output masses to output file
    with open(output_path, 'w') as file:
        for id, mass in zip(ids, masses):
            file.write(f"{id},{'NO CONVERGENCE' if not mass else mass}\n")
