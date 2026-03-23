"""
FOPT for MOA-2
----------------------

Author: Henry Christiansen
Date:   2026-01-26
Email:  hennyc@umich.edu

Description:
    This script runs determines the F10.7 and Ap
    adjustments for a given storm.

    
Usage:
    python fopt.py <Storm> <NRLMSIS Model>
"""

# IMPORTS
import re
import os
import tempfile
import shutil
import json
from matplotlib.patches import Rectangle, Patch
from scipy.ndimage import gaussian_filter1d
import matplotlib.colors as mcolors
from matplotlib.ticker import FixedLocator
from multiprocessing import Pool, cpu_count
from collections import defaultdict
from scipy.stats import linregress
import matplotlib.pyplot as plt
from datetime import datetime
from io import StringIO
from load_gmat import *
from tqdm import tqdm
import pandas as pd
import numpy as np

def read_tle_file(tle_file):
    """
    Docstring for read_tle_file: reads spacecraft TLE data
    from file

    :param tle_file: system path to spacecraft data file
    """
    try: 
        with open(tle_file, 'r') as f:
            tle_data = json.load(f)
        data = defaultdict(list)
        for i in tle_data:
            name = i["NORAD_CAT_ID"]
            data[name].append(i)

        return data
    
    except Exception as e:
        return e

def find_adjustments(args):
    """
    Docstring for find_changes: main function that
    determines both F10.7 and Ap adjustments
    
    :param args: array of inputs
    """
    try:
        # Sorting TLE_DATA
        tle_data, sat_id, mass = args
        epochs = np.array([x['EPOCH'] for x in tle_data], dtype='datetime64')
        ind = np.argsort(epochs)
        sorted_tle_data = [tle_data[i] for i in ind]
        sorted_epochs = epochs[ind]

        _, unique_indices = np.unique(sorted_epochs, return_index=True)
        sorted_tle_data = [sorted_tle_data[i] for i in unique_indices]

        filtered_sorted_tle_data = []
        last_epoch = None
        for tle in sorted_tle_data:
            epoch = np.datetime64(tle['EPOCH'])
            if (START_DATE <= epoch <= END_DATE):
                if last_epoch is None or (epoch - last_epoch) > np.timedelta64(6, 'h'):
                    filtered_sorted_tle_data.append(tle)
                    last_epoch = epoch

        if len(filtered_sorted_tle_data) < 10:
            return False
        
        tle_sma, tle_time = [], []
        start = np.datetime64(filtered_sorted_tle_data[0]['EPOCH'])
        for i in filtered_sorted_tle_data:
            epoch = np.datetime64(i['EPOCH'])
            tle_sma.append(float(i['SEMIMAJOR_AXIS']))
            tle_time.append((epoch - start) / np.timedelta64(1, 'D'))

        tle_dsma = np.gradient(tle_sma, tle_time) * 365

        # Prepare TEMP GMAT SCRIPT and Directory
        path = '/home/hennyc/temp'
        os.makedirs(path, exist_ok=True)
        temp_dir = tempfile.mkdtemp(prefix=f'gmat_{sat_id}_', dir=path)
        script_path = os.path.join(temp_dir, f'gmat_sat_{sat_id}.script')
        weather_path = os.path.join(temp_dir, f'weather_sat_{sat_id}.txt')
        output_path = os.path.join(temp_dir, f'output_{sat_id}.txt')
        
        with open(GMAT_FILE, 'r') as f:
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
            'MASS_VAL': mass,
            'WEATHERFILE_VAL': weather_path,
            'FILENAME_VAL': output_path,
            'LENGTH_VAL': round(((np.datetime64(filtered_sorted_tle_data[-1]['EPOCH']) - np.datetime64
                                (filtered_sorted_tle_data[0]['EPOCH'])) / np.timedelta64(1, 'D')),8)
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
        for i in range(int((END_DATE - START_DATE) / np.timedelta64(1, 'D'))):
            f10_ind = df[df['date'] == str(START_DATE + np.timedelta64(i, 'D') - np.timedelta64(1,'D')).replace('-', ' ')].index[0]
            ap_ind = f10_ind+1
            f10.append(float(content[f10_ind][113:118]))
            start = 47
            for _ in range(8):
                ap.append(int(content[ap_ind][start:start+4]))
                start += 4

        # Solve for f10 changes
        error = None
        last_error = None
        iter = 30
        for i in range(iter):
            rmse, error = find_error(np.concatenate((f10, ap)), script_path, output_path, weather_path, tle_time, tle_dsma, True)
            if isinstance(error, bool):
                return False
            
            if last_error is not None and abs(last_error - rmse) < 1:
                break
            
            f10 = update_f10(f10, error)
            last_error = rmse
        
        # solve for ap changes
        error = None
        last_error = None
        iter2 = 30
        for i in range(iter2):
            rmse, error = find_error(np.concatenate((f10, ap)), script_path, output_path, weather_path, tle_time, tle_dsma, True)
            if isinstance(error, bool):
                return False
            
            if last_error is not None and abs(last_error - rmse) < 1:
                break
            
            ap = update_ap(ap, tle_time, error)
            last_error = rmse

        #print(error)
        shutil.rmtree(temp_dir)
        return np.concatenate((f10, ap))
    except Exception as _:
        shutil.rmtree(temp_dir)
        return False

def find_error(values, script_path, output_path, weather_path, tle_time, tle_dsma, minimize):
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
    update_weather_file(values, weather_path, True)
    
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
            return False
    gmat_dsma = np.array(gmat_dsma) * 365

    if minimize:
        return np.sqrt(np.mean((tle_dsma - gmat_dsma) ** 2)), np.array(tle_dsma - gmat_dsma)
    else:
        return np.array(tle_dsma - gmat_dsma)

def find_window(i, t, tle_time):
    """
    Docstring for find_window: finds time window
    to analyze to determine modeled trajectory 
    dSMA.
    
    :param i: element in tle_time
    :param t: time in tle_time
    :param tle_time: tle_time array
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

def update_weather_file(values, path, subtract):
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

    if subtract:
        f10_ind_for_global = df[df['date'] == str(START_DATE + np.timedelta64(np.argmax(f10_old),'D')).replace('-', ' ')].index[0]
        global_adjustment = np.max(f10_old) - float(content[f10_ind_for_global][113:118])
        f10_ind = df[df['date'] == str(START_DATE - np.timedelta64(2,'D')).replace('-', ' ')].index[0]
        replacement_f10 = float(content[f10_ind][113:118]) + global_adjustment
        replacement_f10 = round(replacement_f10, 1)
        replacement_f10 = np.clip(replacement_f10, 60.0, 400.0)
        content[f10_ind] = content[f10_ind][:113] + f"{replacement_f10:5.1f}" + content[f10_ind][118:]

    for i in range(10):
        if subtract:
            f10_ind = df[df['date'] == str(START_DATE + np.timedelta64(i, 'D') - np.timedelta64(1,'D')).replace('-', ' ')].index[0]
            ap_ind = f10_ind+1
        else:
            f10_ind = df[df['date'] == str(START_DATE + np.timedelta64(i, 'D')).replace('-', ' ')].index[0]
            ap_ind = f10_ind
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

def update_ap(ap, time, error):
    """
    Docstring for update_ap: updates Ap to 
    reduce RMS error
    
    :param ap: current ap values
    :param time: current tle_time array
    :param error: current error array
    """
    time = np.array(time)
    error = np.array(error)

    ap_time = np.linspace(0, 10, len(ap))

    interp_error = np.interp(ap_time, time, error, left=error[0], right=error[-1])

    ap_adjustment = -0.1 * interp_error

    ap_updated = np.clip(ap + ap_adjustment, 0, 400)

    return ap_updated

# def update_ap(ap, time, error):
#     for t,e in zip(time, error):
#         alpha = 0.4
#         beta = 0.4
#         gamma = 0
#         index = min(int(t * 8), 79)
#         if index == 0:
#             ap[index] -= round(alpha*e)
#         elif index == 1:
#             ap[index-1] -= round(beta*e)
#             ap[index] -= round(alpha*e)
#         else:
#             ap[index-2] -= round(gamma*e)
#             ap[index-1] -= round(beta*e)
#             ap[index] -= round(alpha*e)
#     return np.clip(ap, 0, 400)

def update_f10(f10, error):
    """
    Docstring for update_f10: updates
    F10.7 value to reduce RMS error
    
    :param f10: current F10.7 values
    :param error: current error array
    """
    f10 -= 0.1*np.mean(error)
    return np.clip(f10, 60.0, 400.0)

# def update_f10(f10, time, error):
#     i = np.floor(time).astype(int)
#     for start in np.arange(0, 10, 1):
#         alpha = 1
#         indeces = (i >= start) & (i < start+1)
#         if np.any(indeces):
#             selected_errors = error[indeces]
#             if len(selected_errors) == 1:
#                 avg = selected_errors[0]
#             else:
#                 avg = np.median(selected_errors)
#             f10[start] -= avg * alpha
#     f10 = np.round(f10, 1)
#     return np.clip(f10, 60.0, 400.0)

def read_weather_file():
    """
    Docstring for read_weather_file: reads observed space weather data
    """
    with open("/home/hennyc/gmat-git/GMAT/data/atmosphere/earth/SpaceWeather-All-v1.2.txt") as file:
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
    month_dict = {"MAR": "03", "APR": "04", "MAY": "05", "AUG": "08", "SEP": "09", "OCT": "10"}
    m,y = STORM.split('_')
    STORM_FIGURE_STR = f"{month_dict.get(m, None)}-{y}"
    med_vals = np.median(vals, axis=0)
    _, content = read_weather_file()
    f10_original = []
    ap_original = []
    df = pd.read_csv(StringIO(''.join(content)), sep=r'\s+', header=None, low_memory=False)
    df['date'] = df[0].astype(str) + ' ' + df[1].astype(str).str.zfill(2) + ' ' + df[2].astype(str).str.zfill(2)
    for i in range(int((END_DATE - START_DATE) / np.timedelta64(1, 'D'))):
        f10_ind = df[df['date'] == str(START_DATE + np.timedelta64(i, 'D') - np.timedelta64(1,'D')).replace('-', ' ')].index[0]
        ap_ind = f10_ind+1
        f10_original.append(float(content[f10_ind][113:118]))
        start = 47
        for _ in range(8):
            ap_original.append(int(content[ap_ind][start:start+4]))
            start += 4

    vals = np.array(vals)    
    ind_max = np.argmax(f10_original)
    f10_global_adjustments = vals[:, ind_max] - f10_original[ind_max]

# --- Right subplot: Ap heatmap ---
    ap_vals = vals[:, 10:]  # shape (N_spacecraft, 80)
    N_spacecraft, N_ap = ap_vals.shape
    ap_bin_width = 5
    ap_bins = np.arange(0, 400 + ap_bin_width, ap_bin_width)  # y-axis bins
    N_ap_bins = len(ap_bins) - 1

    count_matrix = np.zeros((N_ap_bins, N_ap))
    for col_idx in range(N_ap):
        col_values = ap_vals[:, col_idx]
        counts, _ = np.histogram(col_values, bins=ap_bins)
        count_matrix[:, col_idx] = counts

    # --- Non-linear scaling
    norm = mcolors.PowerNorm(gamma=0.5, vmin=0, vmax=int(np.round(count_matrix[1:-1, :].max() / 10) * 10))

    fig = plt.figure(figsize=(9, 6))
    
    # --- Shared colorbar on the left ---
    cax = fig.add_axes([0.04, 0.1, 0.02, 0.8])  # x, y, width, height
    cb = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap='gray_r'), cax=cax)
    cb.set_label('Number of Spacecraft', fontweight='bold', fontsize=17)
    cb.ax.tick_params(labelsize=17)
    cb.ax.yaxis.set_label_position('left')
    cb.ax.yaxis.set_ticks_position('right')
    # ---- Add "+" to the top tick automatically ----
    ticks = cb.get_ticks()
    labels = [f"{int(t)}" for t in ticks]
    labels[-1] += "+"

    # Fix the locator first
    cb.locator = FixedLocator(ticks)
    cb.set_ticklabels(labels)
    
    # --- Left subplot: F10.7 heatmap ---
    bin_width = 5
    max_abs = np.ceil(np.max(np.abs(f10_global_adjustments)) / bin_width) * bin_width
    max_abs = int(np.round(max_abs / 50) * 50)
    bins = np.arange(-max_abs, max_abs + bin_width, bin_width)
    counts, edges = np.histogram(f10_global_adjustments, bins=bins)
    counts_2d = counts[:, np.newaxis]
    x_edges = np.array([0, 1])
    y_edges = edges

    ax_f10 = fig.add_axes([0.157, 0.1, 0.075, 0.8])
    pcm_f10 = ax_f10.pcolormesh(x_edges, y_edges, counts_2d, cmap='gray_r', norm=norm, shading='auto')

    # y-axis label on left
    ax_f10.yaxis.set_label_position("left")
    ax_f10.tick_params(labelsize=17)
    ax_f10.set_ylabel("F10.7 Global Adjustment (sfu)", fontweight='bold', fontsize=17)

    # y-axis ticks on right
    ax_f10.yaxis.tick_right()
    ax_f10.yaxis.set_ticks_position('right')

    print("F10.7" + str(np.median(f10_global_adjustments)))
    sys.exit()
    ax_f10.axhline(np.median(f10_global_adjustments), color='red', linestyle='-', linewidth=3, label='Adjusted\nF10.7')
    # ax_f10.axhline(0, color='blue', linestyle='-', linewidth=2, label='Adjusted\nF10.7')

    ax_f10.set_xlabel("")
    ax_f10.set_xticks([])
    ax_f10.set_ylim(-max_abs, max_abs)

    # --- Right subplot: Ap heatmap ---
    ax_ap = fig.add_axes([0.363, 0.1, 0.57, 0.8])
    x_edges = np.arange(N_ap + 1)
    y_edges = ap_bins
    pcm_ap = ax_ap.pcolormesh(x_edges, y_edges, count_matrix, cmap='gray_r', norm=norm, shading='auto')

    # # Red lines for original Ap values
    # for i, ap_val in enumerate(ap_original):
    #     if i < N_ap:
    #         ax_ap.plot([i, i + 1], [ap_val, ap_val], color='blue',linestyle='-', linewidth=1.5)

    # ax_ap.plot(np.arange(len(ap_original[:N_ap])) + 0.5, ap_original[:N_ap], color="#29BC00", linestyle='-', linewidth=3, label='Observed $a_p$')
    
    for i, ap_val in enumerate(ap_original):
        if i < N_ap:
            # Find bin index for the Ap value
            bin_idx = np.digitize(ap_val, ap_bins) - 1
            
            # Only draw if inside the valid bin range
            if 0 <= bin_idx < len(ap_bins) - 1:
                y0 = ap_bins[bin_idx]
                y1 = ap_bins[bin_idx + 1]

                # Solid blue rectangle (no transparency)
                ax_ap.add_patch(
                    Rectangle(
                        (i, y0),            # bottom-left corner
                        1,                  # width
                        y1 - y0,            # height
                        facecolor="blue",
                        edgecolor='none',
                        alpha=1.0           # fully opaque
                    )
                )
    
    legend_rect = Patch(
        facecolor="blue",
        edgecolor='none',
        label='Observed $a_p$'
    )

    legend_rect2 = Patch(
        facecolor="red",
        edgecolor='none',
        label='Adjusted $a_p$'
    )
    
    # Step 2: Select the last 80 columns (ap values)
    ap_med_vals = med_vals[10:]  # shape = (80,)

    # Step 3: Percentile and standard deviation
    p75_vals = np.percentile(vals[:, 10:], 75, axis=0)  # shape = (80,)
    std = np.std(ap_med_vals)

    # Step 4: Identify "storm indices" relative to ap_med_vals
    storm_ind = np.where(ap_med_vals > np.median(ap_med_vals) + (2*std))[0]

    # Step 5: Replace storm points with 75th percentile
    ap_med_vals_final = ap_med_vals.copy()
    ap_med_vals_final[storm_ind] = p75_vals[storm_ind]

    for i, ap_val in enumerate(ap_med_vals_final):
        if i < N_ap:
            # Find bin index for the Ap value
            bin_idx = np.digitize(ap_val, ap_bins) - 1
            
            # Only draw if inside the valid bin range
            if 0 <= bin_idx < len(ap_bins) - 1:
                y0 = ap_bins[bin_idx]
                y1 = ap_bins[bin_idx + 1]

                # Solid blue rectangle (no transparency)
                ax_ap.add_patch(
                    Rectangle(
                        (i, y0),            # bottom-left corner
                        1,                  # width
                        y1 - y0,            # height
                        facecolor="red",
                        edgecolor='none',
                        alpha=1.0           # fully opaque
                    )
                )
                
    # ax_ap.plot(np.arange(len(ap_med_vals_final)) + 0.5, ap_med_vals_final, color="#E21717", linestyle='-', linewidth=3, label='Adjusted $a_p$')

    ax_ap.set_xlim(0, N_ap)
    ax_ap.set_ylim(0, 400)
    ax_ap.tick_params(labelsize=17)
    ax_ap.set_xlabel("Real Time", fontweight='bold', fontsize=17)
    ax_ap.set_ylabel(r"3-hr $\mathbf{a_p}$ (nT)", fontweight='bold', fontsize=17)
    ax_ap.yaxis.set_ticks_position('right')
    ax_ap.yaxis.set_label_position("left")
    tick_positions = np.arange(0, N_ap+8, 8)

    # Tick labels as dates starting from START_DATE
    tick_labels = [(START_DATE + np.timedelta64(i, 'D')).astype('M8[D]').astype(str) for i in range(len(tick_positions))]

    # Optional: convert 'YYYY-MM-DD' to 'MM/DD'
    tick_labels = [str(pd.to_datetime(label).strftime('%d')) for label in tick_labels]
    ax_ap.set_xticks(tick_positions)
    ax_ap.set_xticklabels(tick_labels)

    # --- Add title ---
    # ax_f10.legend(loc='upper right', framealpha=1, handlelength=1, fontsize=12)
    ax_ap.legend(handles=[legend_rect, legend_rect2], loc='upper left', frameon=False, fontsize=17)
    fig.suptitle(STORM_FIGURE_STR + r": Global F10.7 and 3-hr $\mathbf{a_p}$ Adjustments of All Spacecraft", fontsize=17, fontweight='bold')

    # plt.show()
    plt.savefig(f'/home/hennyc/data/{STORM}/Results/fopt.png')
    plt.close()

# def find_bias(args):
#     try:
#         # Sorting TLE_DATA
#         tle_data, sat_id, mass = args
#         epochs = np.array([x['EPOCH'] for x in tle_data], dtype='datetime64')
#         ind = np.argsort(epochs)
#         sorted_tle_data = [tle_data[i] for i in ind]
#         sorted_epochs = epochs[ind]

#         _, unique_indices = np.unique(sorted_epochs, return_index=True)
#         sorted_tle_data = [sorted_tle_data[i] for i in unique_indices]

#         filtered_sorted_tle_data = []
#         last_epoch = None
#         for tle in sorted_tle_data:
#             epoch = np.datetime64(tle['EPOCH'])
#             if (START_DATE <= epoch <= END_DATE):
#                 if last_epoch is None or (epoch - last_epoch) > np.timedelta64(6, 'h'):
#                     filtered_sorted_tle_data.append(tle)
#                     last_epoch = epoch

#         if len(filtered_sorted_tle_data) < 10:
#             return False
        
#         tle_sma, tle_time = [], []
#         start = np.datetime64(filtered_sorted_tle_data[0]['EPOCH'])
#         for i in filtered_sorted_tle_data:
#             epoch = np.datetime64(i['EPOCH'])
#             tle_sma.append(float(i['SEMIMAJOR_AXIS']))
#             tle_time.append((epoch - start) / np.timedelta64(1, 'D'))

#         tle_dsma = np.gradient(tle_sma, tle_time) * 365

#         # Prepare TEMP GMAT SCRIPT and Directory
#         path = '/home/hennyc/temp'
#         os.makedirs(path, exist_ok=True)
#         temp_dir = tempfile.mkdtemp(prefix=f'gmat_{sat_id}_', dir=path)
#         script_path = os.path.join(temp_dir, f'gmat_sat_{sat_id}.script')
#         output_path = os.path.join(temp_dir, f'output_{sat_id}.txt')
        
#         with open(GMAT_FILE, 'r') as f:
#             script = f.read()
            
#         epoch = datetime.strptime(filtered_sorted_tle_data[0]['EPOCH'], "%Y-%m-%dT%H:%M:%S.%f")
#         epoch_string = epoch.strftime("%d %b %Y %H:%M:%S.") + f"{epoch.microsecond // 1000:03d}"
#         values = {
#             'EPOCH_VAL': epoch_string,
#             'SMA_VAL': float(filtered_sorted_tle_data[0]['SEMIMAJOR_AXIS']),
#             'ECC_VAL': float(filtered_sorted_tle_data[0]['ECCENTRICITY']),
#             'INC_VAL': float(filtered_sorted_tle_data[0]['INCLINATION']),
#             'RAAN_VAL': float(filtered_sorted_tle_data[0]['RA_OF_ASC_NODE']),
#             'AOP_VAL': float(filtered_sorted_tle_data[0]['ARG_OF_PERICENTER']),
#             'MA_VAL': float(filtered_sorted_tle_data[0]['MEAN_ANOMALY']),
#             'MASS_VAL': mass,
#             'WEATHERFILE_VAL': WEATHER_OUT,
#             'FILENAME_VAL': output_path,
#             'LENGTH_VAL': round(((np.datetime64(filtered_sorted_tle_data[-1]['EPOCH']) - np.datetime64
#                                 (filtered_sorted_tle_data[0]['EPOCH'])) / np.timedelta64(1, 'D')),8)
#         }

#         for key, val in values.items():
#             script = script.replace(key, str(val))
            
#         with open(script_path, 'w') as f:
#             f.write(script)

#         # Solve for bias
#         error = find_bias_error(script_path, output_path, tle_time, tle_dsma)
#         if isinstance(error, bool):
#             return False, False
#         return tle_time, error
    
#     except Exception as e:
#         shutil.rmtree(temp_dir)
#         return False, False

# def find_bias_error(script_path, output_path, tle_time, tle_dsma):
#     try:
#         gmat.LoadScript(script_path)
#         status = gmat.Execute()
#         if status != 1:
#             return False
#     except Exception as e:
#         return False
#     finally:
#         gmat.Clear()

#     try:
#         # Read Data
#         output_data = np.loadtxt(output_path, delimiter=',')
#         time = output_data[:, 0]
#         sma = output_data[:,1]
#         gmat_sma_avg = gaussian_filter1d(sma, sigma=120)
#         gmat_dsma = []
#         for i, t in enumerate(tle_time):
#             t_min, t_max = find_window(i, t, tle_time)
#             mask = (time > t_min) & (time < t_max)
#             if np.sum(mask) >= 2:
#                 x = time[mask]
#                 y = gmat_sma_avg[mask]
#                 slope, *_ = linregress(x, y)
#                 gmat_dsma.append(slope)
#             else:
#                 return False
#         gmat_dsma = np.array(gmat_dsma) * 365
#         return np.array(tle_dsma-gmat_dsma)
#     except Exception as e:
#         return False


if __name__ == '__main__':
    """
    Main execution logic for FOPT.

    """

    if len(sys.argv) != 3:
        print("Usage: python fopt.py <month_year> <NRLMSISE Model>")
        sys.exit(1)

    STORM = sys.argv[1]
    atmospheric_model = sys.argv[2]

    if atmospheric_model not in ("MSISE90", "MSIS21"):
        print("Error: unknown <NRLMSISE Model>. Use MSISE90 or MSIS21")
        print("Usage: python fopt.py <month_year> <NRLMSISE Model>")
        sys.exit(1)

    with open(f"/home/hennyc/data/{STORM}/DATES.txt", "r") as file:
        _, _, _ = file.readline().strip().split(",")
        _, START_STR_2, END_STRING_2 = file.readline().strip().split(",")
        
    START_DATE, END_DATE = np.datetime64(START_STR_2), np.datetime64(END_STRING_2)
    base = f"/home/hennyc/data/{STORM}"
    TLE_FILE = f"{base}/TLE_DATA.json"
    MASS_FILE = f"{base}/MOPT_OUTPUT.txt"
    WEATHER_OUT = f"{base}/WEATHER.txt"
    GMAT_FILE = "/home/hennyc/src/fopt.script"

    # Modify mopt.script with inputted atmospheric model
    with open(GMAT_FILE, 'r') as f:
        script = f.read()
    script = re.sub(r"\b(MSISE90|MSIS21)\b", "ATM_MOD_VAL", script)
    script = script.replace("ATM_MOD_VAL", str(atmospheric_model))
    with open(GMAT_FILE, 'w') as f:
        f.write(script)

    # clean_vals = np.load(f'/home/hennyc/data/{STORM}/Results/clean_vals_{STORM}.npy')
    # plot_ap_f10(clean_vals)
    # sys.exit()

    data = read_tle_file(TLE_FILE)

    if isinstance(data, BaseException):
        print(f"Problem in read_tle_file(): {data}")

    mass = {}
    with open(MASS_FILE, "r") as f:
        for line in f:
            id, m = line.strip().split(",")
            id = id.strip()
            m = m.strip()

            if m != "NO CONVERGENCE":
                mass[id] = float(m)
    
    ids = list(mass.keys())
    args_list = [(data[sat_id], sat_id, mass[sat_id]) for sat_id in ids]

    with Pool(cpu_count()) as pool:
        vals = list(tqdm(pool.imap(find_adjustments, args_list), total=len(args_list), desc="FOPT"))
    
    clean_vals = [v for v in vals if not isinstance(v, bool) and v is not False]
    np.save(f'/home/hennyc/data/{STORM}/clean_vals_{STORM}.npy', np.array(clean_vals))
    print(str(len(clean_vals)) + "/" + str(len(vals)) + " converged")

    med_vals = np.median(clean_vals, axis=0)
    p75_vals = np.percentile(clean_vals, 75, axis=0)
    std = np.std(med_vals[10:])
    storm_ind = np.where(med_vals[10:] > np.median(med_vals[10:])+(std*2))[0] + 10
    # peak_idx = np.argmax(med_vals)
    # storm_ind = np.arange(max(0, peak_idx - 1), peak_idx + 1)
    med_vals[storm_ind] = p75_vals[storm_ind]
    print(storm_ind)

    h, c = read_weather_file()
    write_weather_file(WEATHER_OUT, h, c)
    update_weather_file(med_vals, WEATHER_OUT, True)

    plot_ap_f10(clean_vals)


    # Find any bias
    #with Pool(cpu_count()) as pool:
    #    bias = list(tqdm(pool.imap(find_bias, args_list), total=len(args_list), desc="Bias"))

    #plt.figure(figsize=(10, 6))

    #for _, (tle_time, error) in enumerate(bias):
    #    if isinstance(tle_time, bool) or isinstance(error, bool):
    #        continue
    #    plt.plot(tle_time, error, 'ko-', color='black')

    #plt.xlabel('Time')
    #plt.ylabel('Absolute Error')
    #plt.title('Errors after median f10 and ap adjustments for all sats')
    #plt.tight_layout()
    #plt.savefig(f'../STORMS/{STORM}/Results/bias.png')
