"""
FOPT for MOA
----------------------

Author: Henry Christiansen
Date:   2026-07-21
Email:  hennyc@umich.edu

Description:
    This script runs determines the F10.7 and Ap
    adjustments for a given date range.

Usage:
    python fopt.py <Storm> <NRLMSIS Model>
"""

# IMPORTS
import re
import os
import tempfile
import shutil
import json
import argparse
from datetime import timedelta
import matplotlib as mpl
from scipy.signal import savgol_filter
from matplotlib.ticker import MaxNLocator
from matplotlib.patches import Rectangle, Patch
from scipy.interpolate import UnivariateSpline
import matplotlib.colors as mcolors
from matplotlib.ticker import FixedLocator
from multiprocessing import Pool, cpu_count
from collections import defaultdict
import matplotlib.pyplot as plt
from datetime import datetime
from io import StringIO
from load_gmat import *
from tqdm import tqdm
import pandas as pd
import numpy as np

def get_tle_data(file, sat_ids):
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

    start_date = pd.to_datetime(START_DATE)
    end_date = pd.to_datetime(END_DATE)

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

def get_derivative(x_arr, y_arr):
    x_mid = (x_arr[:-1] + x_arr[1:]) / 2
    dsma = (y_arr[1:] - y_arr[:-1]) / (x_arr[1:] - x_arr[:-1])
    return np.array(x_mid), np.array(dsma)

def find_adjustments(args):
    """
    Docstring for find_changes: main function that
    determines both F10.7 and Ap adjustments
    
    :param args: array of inputs
    """
    try:
        # Sorting TLE_DATA
        tle_data, sat_id, mass = args

        start_date = pd.to_datetime(START_DATE)
        end_date = pd.to_datetime(END_DATE)
        
        tle_sma, tle_sma_t = [], []

        for entry in tle_data:
            epoch = pd.to_datetime(entry['EPOCH'])
            tle_sma_t.append((epoch - start_date).total_seconds() / 86400.0)
            tle_sma.append(float(entry["SEMIMAJOR_AXIS"]))
            if not start_date <= epoch <= end_date:
                print("TLE DATA out of range - shouldn't happen")
                sys.exit()

        tle_sma_t = np.array(tle_sma_t)
        tle_sma = np.array(tle_sma)

        tle_dsma_t, tle_dsma = get_derivative(tle_sma_t, tle_sma)
        tle_dsma_t, tle_dsma = get_spline(tle_dsma_t, tle_dsma)

        if np.any(tle_dsma > 0):
            return False

        tle_dsma = tle_dsma * 1000

        # Prepare TEMP GMAT SCRIPT and Directory
        path = './temp'
        os.makedirs(path, exist_ok=True)
        temp_dir = tempfile.mkdtemp(prefix=f'gmat_{sat_id}_', dir=path)
        script_path = os.path.join(temp_dir, f'gmat_sat_{sat_id}.script')
        weather_path = os.path.join(temp_dir, f'weather_sat_{sat_id}.txt')
        output_path = os.path.join(temp_dir, f'output_{sat_id}.txt')
        
        with open(GMAT_FILE, 'r') as f:
            script = f.read()
            
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
            'MASS_VAL': mass,
            'WEATHERFILE_VAL': weather_path,
            'FILENAME_VAL': output_path,
            'LENGTH_VAL': round((tle_sma_t[-1] - tle_sma_t[0]),8)
        }

        for key, val in values.items():
            script = script.replace(key, str(val))
            
        with open(script_path, 'w') as f:
            f.write(script)

        # Get initial weather
        _, content = read_weather_file()
        f10 = []
        ap = []
        df = pd.read_csv(StringIO(''.join(content)), sep=r'\s+', header=None, low_memory=False)
        df['date'] = df[0].astype(str) + ' ' + df[1].astype(str).str.zfill(2) + ' ' + df[2].astype(str).str.zfill(2)
        for i in range(10):
            f10_ind = df[df['date'] == str(START_DATE + np.timedelta64(i, 'D') - np.timedelta64(1,'D')).replace('-', ' ')].index[0]
            ap_ind = f10_ind+1
            f10.append(float(content[f10_ind][113:118]))
            start = 47
            for _ in range(8):
                ap.append(int(content[ap_ind][start:start+4]))
                start += 4

        # Solve for f10 changes
        errors_f10 = []
        error = None
        last_error = None
        iter1 = 50
        orig_rmse = None
        last_vals = None

        for i in range(iter1):

            rmse, error = find_error(np.concatenate((f10, ap)), script_path, output_path, weather_path, tle_sma_t, tle_sma, tle_dsma_t, tle_dsma, False)
            
            if isinstance(error, bool):
                return False

            if i == 0:
                orig_rmse = rmse
                errors_f10.append(1)
            else:
                errors_f10.append(rmse/orig_rmse)
            
            if last_error is not None and last_error - rmse/orig_rmse < 0.01:
                if last_error - rmse/orig_rmse < 0:
                    f10 = last_vals[:10]
                    ap = last_vals[10:]
                    errors_f10.pop(-1)
                break

            last_vals = np.concatenate((f10, ap))
            
            f10 = update_f10(f10, tle_dsma_t, error)
            last_error = rmse/orig_rmse

        if len(errors_f10) < iter1:
            errors_f10.extend([np.nan] * (iter1 - len(errors_f10)))

        # solve for ap changes
        errors_ap = []
        error = None
        last_error = None
        iter2 = 50

        for i in range(iter2):
            rmse, error = find_error(np.concatenate((f10, ap)), script_path, output_path, weather_path, tle_sma_t, tle_sma, tle_dsma_t, tle_dsma, False)

            if isinstance(error, bool):
                return False

            errors_ap.append(rmse/orig_rmse)

            if last_error is not None and last_error - rmse/orig_rmse < 0.01:
                if last_error - rmse/orig_rmse < 0:
                    f10 = last_vals[:10]
                    ap = last_vals[10:]
                    errors_ap.pop(-1)
                break

            last_vals = np.concatenate((f10, ap))

            ap, f10 = update_ap(ap, f10, tle_dsma_t, error)
            last_error = rmse/orig_rmse

        if len(errors_ap) < iter2:
            errors_ap.extend([np.nan] * (iter2 - len(errors_ap)))  

        interp_bias = np.interp(np.arange(0, 10, 0.01), np.arange(tle_dsma_t[0], tle_dsma_t[-1], 0.01), error, left=error[0], right=error[-1])    

        #print(error)
        shutil.rmtree(temp_dir)
        return np.concatenate((f10, ap)), {"f10": errors_f10, "ap": errors_ap, "bias": interp_bias}
    except Exception as _:
        shutil.rmtree(temp_dir)
        return False

def find_error(values, script_path, output_path, weather_path, tle_sma_t, tle_sma, tle_dsma_t, tle_dsma, plot):
    """
    Docstring for find_error: function to determine the RMS
    error between a modeled GMAT trajectory and the 
    spacecraft TLE data.
    
    :param values: weather_file data values
    :param script_path: path to fopt script template
    :param output_path: path to GMAT output file
    :param weather_path: path to weather file
    :param tle_time: array of times of TLEs
    :param tle_dsma: array of dSMAs of TLEs
    :param minimize: bool determining if function 
    should minimize the mass 
    """
    # Update weather file
    update_weather_file(values, weather_path)
    
    # Run GMAT
    try:
        gmat.LoadScript(script_path)
        status = gmat.Execute()
        if status != 1:
            return False
    except Exception as e:
        return False
    finally:
        gmat.Clear()

    output_data = np.loadtxt(output_path, delimiter=',')
    time = np.array(output_data[:, 0]) + tle_sma_t[0]
    sma = np.array(output_data[:,1])
    ma = np.mod(output_data[:, 2], 360)

    wraps = np.diff(ma) < -300 
    orbit_id = np.cumsum(np.insert(wraps, 0, False))
    counts = np.bincount(orbit_id)
    sma_orbit_avg = np.bincount(orbit_id, weights=sma) / counts
    time_orbit_avg = np.bincount(orbit_id, weights=time) / counts

    k = 5
    n = len(sma_orbit_avg) // k  

    sma_orbit_avg = sma_orbit_avg[:n*k].reshape(n, k).mean(axis=1)
    time_orbit_avg = time_orbit_avg[:n*k].reshape(n, k).mean(axis=1)

    gmat_spline = UnivariateSpline(time_orbit_avg, sma_orbit_avg, s = 0)
    gmat_times = np.arange(tle_dsma_t[0], tle_dsma_t[-1], 0.01)
    gmat_dsma = gmat_spline.derivative()(tle_dsma_t) * 1000

    if plot:
        plt.plot(tle_dsma_t, tle_dsma, 'k-')
        plt.plot(tle_dsma_t, gmat_dsma, 'r-')
        plt.show()
        plt.close()

    assert len(tle_dsma) == len(gmat_dsma)
    return np.sqrt(np.mean((tle_dsma - gmat_dsma) ** 2)), np.array(tle_dsma - gmat_dsma)

def update_weather_file(values, path):
    """
    Docstring for update_weather_file
    
    :param values: data of weather file
    :param path: path to weather file
    :param subtract: bool whether to move F10.7 daily
    values back one day (as MSIS uses previous day)
    """
    assert len(values) == 90

    f10_old = values[:10]
    ap_old = [values[10 + i*8 : 10 + (i+1)*8] for i in range(10)]

    header, content = read_weather_file()

    df = pd.read_csv(StringIO(''.join(content)), sep=r'\s+', header=None, low_memory=False)
    df['date'] = df[0].astype(str) + ' ' + df[1].astype(str).str.zfill(2) + ' ' + df[2].astype(str).str.zfill(2)

    f10_ind = df[df['date'] == str(START_DATE - np.timedelta64(2,'D')).replace('-', ' ')].index[0]
    content[f10_ind] = content[f10_ind][:113] + f"{round(f10_old[0], 1):5.1f}" + content[f10_ind][118:]

    f10_ind = df[df['date'] == str(START_DATE + np.timedelta64(9,'D')).replace('-', ' ')].index[0]
    content[f10_ind] = content[f10_ind][:113] + f"{round(f10_old[-1], 1):5.1f}" + content[f10_ind][118:]

    for i in range(10):
        f10_ind = df[df['date'] == str(START_DATE + np.timedelta64(i, 'D') - np.timedelta64(1,'D')).replace('-', ' ')].index[0]
        ap_ind = f10_ind+1
        content[f10_ind] = content[f10_ind][:113] + f"{round(f10_old[i], 1):5.1f}" + content[f10_ind][118:]
        Ap = [int(round(v)) for v in ap_old[i]]
        Ap.append(int(round(sum(Ap) / 8))) 
        start = 47
        for ap in Ap:
            content[ap_ind] = content[ap_ind][:start] + f"{ap:>3}" + content[ap_ind][start+3:]
            start += 4

    write_weather_file(path, header, content)

def write_weather_file(file, header, content):
    """
    Docstring for write_weather_file: writes weather file
    
    :param file: system file path to weather file
    :param header: weather file header data
    :param content: weather file data
    """
    with open(file, 'w') as file:
        file.writelines(header)
        file.writelines(content) 

def update_ap(ap, f10, tle_dsma_t, error):
    """

    """
    tle_dsma_t = np.array(tle_dsma_t)
    error = np.array(error)

    ap_time = np.linspace(0.125/2, 10 - 0.125/2, 80)
    error_time = np.arange(tle_dsma_t[0], tle_dsma_t[-1], 0.01)
    assert len(error_time) == len(error)
    
    interp_error = np.interp(ap_time, error_time, error, left=error[0], right=error[-1])
    
    adjustment = np.round(-0.1 * interp_error)

    ap = ap + adjustment

    ap = redistribute_ap(ap)

    for i in range(10):
        chunk = ap[i*8:(i+1)*8]
        if (chunk < 1).all():
            f10[i] -= 5

    ap = np.clip(ap, 0, 400)
    f10 = np.clip(f10, 60.0, 400.0)

    return ap, f10

def redistribute_ap(ap):
    ap = ap.copy()
    limit = 400
    for i in range(len(ap)):
        if ap[i] > limit:
            excess = ap[i] - limit
            ap[i] = limit

            j = i - 1
            while excess > 0 and j >= 0:
                space = limit - ap[j]

                if space > 0:
                    transfer = min(space, excess)
                    ap[j] += transfer
                    excess -= transfer
                j -= 1
    return ap

def update_f10(f10, tle_dsma_t, error):
    """
    Docstring for update_f10: updates
    F10.7 value to reduce RMS error
    
    :param f10: current F10.7 values
    :param error: current error array
    """

    f10 -= 0.1*np.median(error)
    return np.clip(f10, 60.0, 400.0)

def read_weather_file():
    """
    Docstring for read_weather_file: reads observed space weather data
    """
    with open(f"{GMAT_PATH}/data/atmosphere/earth/SpaceWeather-All-v1.2.txt") as file:
        lines = file.readlines()
    
    header_end = next(i for i, line in enumerate(lines) if 'BEGIN OBSERVED' in line)
    header = lines[0:header_end+1]
    content = lines[header_end + 1:]
    return header, content

def plot_ap_f10(vals):
    """
    Docstring for plot_ap_f10: plots FOPT results
    
    :param vals: Final adjusted F10.7 and ap values
    """
    med_vals = np.median(vals, axis=0)
    _, content = read_weather_file()
    f10_original = []
    ap_original = []
    df = pd.read_csv(StringIO(''.join(content)), sep=r'\s+', header=None, low_memory=False)
    df['date'] = df[0].astype(str) + ' ' + df[1].astype(str).str.zfill(2) + ' ' + df[2].astype(str).str.zfill(2)
    for i in range(10):
        f10_ind = df[df['date'] == str(START_DATE + np.timedelta64(i, 'D') - np.timedelta64(1,'D')).replace('-', ' ')].index[0]
        ap_ind = f10_ind+1
        f10_original.append(float(content[f10_ind][113:118]))
        start = 47
        for _ in range(8):
            ap_original.append(int(content[ap_ind][start:start+4]))
            start += 4

    vals = np.array(vals)    

    ap_vals = vals[:, 10:]  
    N_spacecraft, N_ap = ap_vals.shape
    ap_bin_width = 10
    ap_bins = np.arange(0, 400 + ap_bin_width, ap_bin_width) 
    N_ap_bins = len(ap_bins) - 1

    count_matrix = np.zeros((N_ap_bins, N_ap))
    for col_idx in range(N_ap):
        col_values = ap_vals[:, col_idx]
        counts, _ = np.histogram(col_values, bins=ap_bins)
        count_matrix[:, col_idx] = counts

    norm = mcolors.PowerNorm(gamma=1, vmin=0, vmax=int(np.round(count_matrix[1:-1, :].max() / 10) * 10))

    fig = plt.figure(figsize=(9, 6))
    size = 14
    
    cax = fig.add_axes([0.04, 0.1, 0.02, 0.87])  # x, y, width, height
    cb = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap='gray_r'), cax=cax)
    cb.set_label('Number of Spacecraft', fontweight='bold', fontsize=size)
    cb.ax.tick_params(labelsize=size)
    cb.ax.yaxis.set_label_position('left')
    cb.ax.yaxis.set_ticks_position('right')
    ticks = cb.get_ticks()
    labels = [f"{int(t)}" for t in ticks]
    labels[-1] += "+"

    cb.locator = FixedLocator(ticks)
    cb.set_ticklabels(labels)

    ax_ap = fig.add_axes([0.154, 0.1, 0.785, 0.87])
    x_edges = np.arange(N_ap + 1)
    y_edges = ap_bins
    pcm_ap = ax_ap.pcolormesh(x_edges, y_edges, count_matrix, cmap='gray_r', norm=norm, shading='auto', zorder=1)

    ax_ap.plot(np.arange(len(ap_original[:N_ap])) + 0.5, ap_original[:N_ap], color="blue", linestyle='-', linewidth=3, label='Observed $a_p$')
    
    
    legend_rect = Patch(
        facecolor="blue",
        edgecolor='none',
        label='Observed $a_p$'
    )

    legend_rect2 = Patch(
        facecolor="red",
        edgecolor='none',
        label='Median Adjusted $a_p$'
    )
    
    ap_med_vals = med_vals[10:]  # shape = (80,)
    ax_ap.plot(np.arange(len(ap_med_vals)) + 0.5, ap_med_vals, color="red", linestyle='-', linewidth=3, label='Median Adjusted $a_p$')
    ax_ap.set_xlim(0, N_ap)
    ax_ap.set_ylim(0, 400)
    ax_ap.tick_params(labelsize=size)
    ax_ap.set_xlabel("Real Time", fontweight='bold', fontsize=size)
    ax_ap.set_ylabel(r"3-hr $\mathbf{a_p}$ (nT)", fontweight='bold', fontsize=size)
    ax_ap.yaxis.set_ticks_position('right')
    ax_ap.yaxis.set_label_position("left")
    tick_positions = np.arange(0, N_ap+8, 8)
    tick_labels = [(START_DATE + np.timedelta64(i, 'D')).astype('M8[D]').astype(str) for i in range(len(tick_positions))]
    tick_labels = [str(pd.to_datetime(label).strftime('%m-%d')) for label in tick_labels]
    ax_ap.set_xticks(tick_positions)
    ax_ap.set_xticklabels(tick_labels)
    ax_ap.legend(handles=[legend_rect, legend_rect2], loc='upper left', frameon=False, fontsize=size)
    # fig.suptitle(STORM_FIGURE_STR + r" Storm: 3-hr $\mathbf{a_p}$ Adjustments of all Spacecraft", fontsize=size, fontweight='bold')
    
    plt.savefig(f'{BASE_PATH}/Results/fopt_ap.png', dpi=300)
    # plt.show()
    plt.close()


    # F10.7 heatmap
    f10_vals = vals[:, :10]  
    N_spacecraft, N_f10 = f10_vals.shape
    f10_bin_width = 5
    f10_bins = np.arange(60.0, 400.0 + f10_bin_width, f10_bin_width)  # y-axis bins
    N_f10_bins = len(f10_bins) - 1

    count_matrix = np.zeros((N_f10_bins, N_f10))
    for col_idx in range(N_f10):
        col_values = f10_vals[:, col_idx]
        counts, _ = np.histogram(col_values, bins=f10_bins)
        count_matrix[:, col_idx] = counts

    norm = mcolors.PowerNorm(gamma=1, vmin=0, vmax=int(np.round(count_matrix[1:-1, :].max() / 10) * 10))

    fig = plt.figure(figsize=(9, 6))
    size = 14
    cax = fig.add_axes([0.04, 0.1, 0.02, 0.87])  # x, y, width, height
    cb = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap='gray_r'), cax=cax)
    cb.set_label('Number of Spacecraft', fontweight='bold', fontsize=size)
    cb.ax.tick_params(labelsize=size)
    cb.ax.yaxis.set_label_position('left')
    cb.ax.yaxis.set_ticks_position('right')
    ticks = cb.get_ticks()
    labels = [f"{int(t)}" for t in ticks]
    labels[-1] += "+"
  
    cb.locator = FixedLocator(ticks)
    cb.set_ticklabels(labels)
  
    ax_ap = fig.add_axes([0.154, 0.1, 0.785, 0.87])
    x_edges = np.arange(N_f10 + 1)
    y_edges = f10_bins
    pcm_ap = ax_ap.pcolormesh(x_edges, y_edges, count_matrix, cmap='gray_r', norm=norm, shading='auto', zorder=1)

    ax_ap.plot(np.arange(len(f10_original[:N_f10])) + 0.5, f10_original[:N_f10], color="blue", linestyle='-', linewidth=3, label='Observed $a_p$')
    
    legend_rect = Patch(
        facecolor="blue",
        edgecolor='none',
        label='Observed F10.7'
    )

    legend_rect2 = Patch(
        facecolor="red",
        edgecolor='none',
        label='Median Adjusted F10.7'
    )
    
    f10_med_vals = med_vals[:10]  # shape = (80,)
    ax_ap.plot(np.arange(len(f10_med_vals)) + 0.5, f10_med_vals, color="red", linestyle='-', linewidth=3, label='Median Adjusted F10.7')
    ax_ap.set_xlim(0, N_f10)
    ax_ap.set_ylim(
        np.minimum(np.min(f10_vals), np.min(f10_original[:N_f10])) - 10,
        np.maximum(np.max(f10_vals), np.max(f10_original[:N_f10])) + 20
    )
    ax_ap.tick_params(labelsize=size)
    ax_ap.set_xlabel("Real Time", fontweight='bold', fontsize=size)
    ax_ap.set_ylabel("F10.7 (sfu)", fontweight='bold', fontsize=size)
    ax_ap.yaxis.set_ticks_position('right')
    ax_ap.yaxis.set_label_position("left")
    tick_positions = np.arange(0, 11, 1)
    tick_labels = [(START_DATE + np.timedelta64(i, 'D')).astype('M8[D]').astype(str) for i in range(len(tick_positions))]
    tick_labels = [str(pd.to_datetime(label).strftime('%m-%d')) for label in tick_labels]
    ax_ap.set_xticks(tick_positions)
    ax_ap.set_xticklabels(tick_labels)
    ax_ap.legend(handles=[legend_rect, legend_rect2], loc='upper left', frameon=False, fontsize=size)
    # fig.suptitle(STORM_FIGURE_STR + r" Storm: Daily F10.7 Adjustments of all Spacecraft", fontsize=size, fontweight='bold')
    plt.savefig(f'{BASE_PATH}/Results/fopt_f10.png', dpi=300)
    # plt.show()
    plt.close()

def plot_convergence(f10_vals, ap_vals):

    if len(f10_vals) == 0 and len(ap_vals) == 0:
        print("No valid data to plot.")
        return
    
    mpl.rcParams.update({
        "font.size": 14,
        "axes.labelsize": 14,
        "axes.titlesize": 14,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "ytick.labelsize": 14
    })

    fig, axes = plt.subplots(1, 2, figsize=(9, 6), sharey=True)

    if len(f10_vals) > 0:
        f10_data = np.array(f10_vals)
        f10_iters = np.arange(f10_data.shape[1])

        for row in f10_data:
            axes[0].plot(f10_iters, row, color='black', alpha=1)

    axes[0].set_title("FOPT", fontweight='bold')
    axes[0].set_xlabel("Step", fontweight='bold')

    if len(ap_vals) > 0:
        ap_data = np.array(ap_vals)
        ap_iters = np.arange(ap_data.shape[1])

        for row in ap_data:
            axes[1].plot(ap_iters, row, color='black', alpha=1)

    axes[1].set_title("AOPT", fontweight='bold')
    axes[1].set_xlabel("Step", fontweight='bold')

    fig.supylabel("Normalized RMSE", fontweight='bold')
    # fig.suptitle(f"{STORM_FIGURE_STR} Storm: RMSE Reduction of all Spacecraft", fontweight='bold')
    axes[0].xaxis.set_major_locator(MaxNLocator(integer=True))
    axes[1].xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.tight_layout()
    plt.savefig(f'{BASE_PATH}/Results/fopt_convergence.png', dpi=300)
    # plt.show()

def plot_bias(b):
    plt.figure(figsize=(8, 5))

    for i in range(len(b)):
        plt.plot(np.arange(0, 10, 0.01), b[i], alpha=0.7)
    plt.xlim(0, 10)
    plt.xlabel("Time")
    plt.ylabel("Error")
    plt.title("Residual Errors after FOPT/AOPT")
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{BASE_PATH}/Results/errors.png')
    # plt.show()

def get_spline(x, y):
    y_min = np.min(y)
    y_max = np.max(y)
    y_norm = (y - y_min) / (y_max - y_min)

    median = np.median(y_norm)
    mad = np.median(np.abs(y_norm - median))  
    sigma_robust = 1.4826 * mad   

    mean = np.mean(y_norm)
    w = np.ones_like(y_norm) * (1.0 / sigma_robust)
    neg_outliers = y_norm < (mean - 2 * sigma_robust)
    w[neg_outliers] *= 1.0   
    spline = UnivariateSpline(x, y_norm, w=w, s=len(x))
    spline_zero = UnivariateSpline(x, y_norm, s = 0)
    
    x_ret = np.arange(x[0], x[-1], 0.01)
    y_ret1 = spline(x_ret)
    y_ret2 = spline_zero(x_ret)

    y_ret = (0.5*y_ret1 + 0.5*y_ret2)

    return x_ret, y_ret2 * (y_max - y_min) + y_min

if __name__ == '__main__':
    """
    Main execution logic for FOPT.

    """

    parser = argparse.ArgumentParser()
    parser.add_argument("date")
    parser.add_argument("model")
    parser.add_argument("src_path")
    parser.add_argument("data_path")
    parser.add_argument("gmat_path")
    args = parser.parse_args()
    STORM = args.date
    atmospheric_model = args.model

    src_path = args.src_path
    data_path = args.data_path
    GMAT_PATH = args.gmat_path

    if atmospheric_model not in ("MSISE90", "NRLMSISE00"):
        print("Error: unknown <NRLMSISE Model>. Use MSISE90, NRLMSISE00")
        print("Usage: python fopt.py <name> <NRLMSISE Model>")
        sys.exit(1)

    date_file = f"{data_path}/{STORM}/DATES.txt"

    with open(date_file, "r") as file:
        _, _, _ = file.readline().strip().split(",")
        _, START_STR_2, END_STRING_2 = file.readline().strip().split(",")
        
    START_DATE, END_DATE = np.datetime64(START_STR_2), np.datetime64(END_STRING_2)
    
    BASE_PATH = f"/home/hennyc/afrl/moa/{STORM}"

    TLE_FILE = f"{BASE_PATH}/TLE_DATA_FOPT.json"
    MASS_FILE = f"{BASE_PATH}/MOPT_OUTPUT.txt"
    WEATHER_OUT = f"{BASE_PATH}/WEATHER.txt"
    GMAT_FILE = f"{src_path}/fopt.script"

    STORM_FIGURE_STR = STORM

    with open(GMAT_FILE, 'r') as f:
        script = f.read()
    script = re.sub(r"\b(MSISE90|NRLMSISE00|MSIS21)\b", "ATM_MOD_VAL", script)
    script = script.replace("ATM_MOD_VAL", str(atmospheric_model))
    with open(GMAT_FILE, 'w') as f:
        f.write(script)

    # clean_vals = np.load(f'{BASE_PATH}/Results/fopt_adjustments.npy')
    # plot_ap_f10(clean_vals)
    # sys.exit()

    mass = {}
    with open(MASS_FILE, "r") as f:
        for line in f:
            id, m = line.strip().split(",")
            id = id.strip()
            m = m.strip()

            mass[id] = float(m)
    
    ids = list(mass.keys())
    data = get_tle_data(TLE_FILE, ids)

    if isinstance(data, BaseException):
        print(f"Problem in read_tle_file(): {data}")
    
    args_list = [(data[sat_id], sat_id, mass[sat_id]) for sat_id in ids]

    with Pool(cpu_count()) as pool:
        vals = list(tqdm(pool.imap(find_adjustments, args_list), total=len(args_list), desc="FOPT"))
    
    clean_vals = [v for v in vals if v is not False]

    adjustments = []
    errors_f10 = []
    errors_ap = []
    biases = []

    for adj, err_dict in clean_vals:
        adjustments.append(adj)
        errors_f10.append(err_dict["f10"])
        errors_ap.append(err_dict["ap"])
        biases.append(err_dict["bias"])
    
    adjustments = np.array(adjustments)
    errors_f10 = np.array(errors_f10)
    errors_ap = np.array(errors_ap)
    biases = np.array(biases)

    np.save(f'{BASE_PATH}/Results/fopt_adjustments.npy', adjustments)
    print(str(len(clean_vals)) + "/" + str(len(vals)) + " converged")

    final_error = np.array([
        err[np.isfinite(err)][-1] if np.any(np.isfinite(err)) else np.nan
        for err in errors_ap])
    
    weighted_mean_vals = np.average(adjustments, axis=0, weights= 1 / final_error)


    med_vals = np.median(adjustments, axis=0)
    mean_vals = np.mean(adjustments, axis=0)
    mean_vals_upper = np.percentile(adjustments, q=75, axis=0)
    mean_vals_lower = np.percentile(adjustments, q=25, axis=0)

    h, c = read_weather_file()
    write_weather_file(WEATHER_OUT, h, c)
    update_weather_file(med_vals, WEATHER_OUT)

    write_weather_file(f"{BASE_PATH}/WEATHER_WEIGHT_MEAN.txt", h, c)
    update_weather_file(weighted_mean_vals, f"{BASE_PATH}/WEATHER_WEIGHT_MEAN.txt")

    write_weather_file(f"{BASE_PATH}/WEATHER_MEAN.txt", h, c)
    update_weather_file(mean_vals, f"{BASE_PATH}/WEATHER_MEAN.txt")

    write_weather_file(f"{BASE_PATH}/WEATHER_MEAN_UPPER.txt", h, c)
    update_weather_file(mean_vals_upper, f"{BASE_PATH}/WEATHER_MEAN_UPPER.txt")

    write_weather_file(f"{BASE_PATH}/WEATHER_MEAN_LOWER.txt", h, c)
    update_weather_file(mean_vals_lower, f"{BASE_PATH}/WEATHER_MEAN_LOWER.txt")

    plot_ap_f10(adjustments)
    plot_convergence(errors_f10, errors_ap)
    plot_bias(biases)
