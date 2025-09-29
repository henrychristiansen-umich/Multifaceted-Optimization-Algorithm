# FOPT solver for MOA 2
#
# Coded by Henry Christiansen. University of Michigan.
#
# This file solves for the F10.7 and 3 hour ap adjustments 
# that minimize the rms difference between GMAT and TLE data

# IMPORTS
from scipy.ndimage import gaussian_filter1d
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
import os
import tempfile
import shutil
import json

def read_tle_file():
    try: 
        with open(TLE_FILE, 'r') as f:
            tle_data = json.load(f)
        data = defaultdict(list)
        for i in tle_data:
            name = i["NORAD_CAT_ID"]
            data[name].append(i)

        return data
    
    except Exception as e:
        return e

def find_changes(args):
    try:
        # Sorting TLE_DATA
        tle_data, sat_id, mass = args
        epochs = np.array([x['EPOCH'] for x in tle_data], dtype='datetime64')
        ind = np.argsort(epochs)
        sorted_tle_data = [tle_data[i] for i in ind]
        sorted_epochs = epochs[ind]
        _, unique_indices = np.unique(sorted_epochs, return_index=True)
        sorted_tle_data = [sorted_tle_data[i] for i in unique_indices]

        tle_sma, tle_time, filtered_sorted_tle_data = [], [], []
        for i in sorted_tle_data:
            epoch = np.datetime64(i['EPOCH'])
            if (START_DATE <= epoch <= END_DATE):
                tle_sma.append(float(i['SEMIMAJOR_AXIS']))
                filtered_sorted_tle_data.append(i)
        
        start = np.datetime64(filtered_sorted_tle_data[0]['EPOCH'])
        for i in filtered_sorted_tle_data:
            epoch = np.datetime64(i['EPOCH'])
            tle_time.append((epoch - start) / np.timedelta64(1, 'D'))

        if len(filtered_sorted_tle_data) < 2:
            return False

        tle_dsma = np.gradient(tle_sma, tle_time) * 365 #km/year

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
        for _ in range(30):
            error = find_error(np.concatenate((f10, ap)), script_path, output_path, weather_path, tle_time, tle_dsma, False)
            if isinstance(error, bool):
                return False

            if last_error is not None:
                if abs(np.sqrt(np.mean(last_error**2)) - np.sqrt(np.mean(error**2))) < 0.01:
                    break

            f10 = update_f10(f10, error)
            last_error = error
        
        # solve for ap changes
        error = None
        last_error = None
        for _ in range(30):
            error = find_error(np.concatenate((f10, ap)), script_path, output_path, weather_path, tle_time, tle_dsma, False)
            if isinstance(error, bool):
                return False

            if last_error is not None:
                if abs(np.sqrt(np.mean(last_error**2)) - np.sqrt(np.mean(error**2))) < 0.01:
                    break

            ap = update_ap(ap, tle_time, error)
            last_error = error

        shutil.rmtree(temp_dir)
        return np.concatenate((f10, ap))
    except Exception as _:
        shutil.rmtree(temp_dir)
        return False

def find_error(values, script_path, output_path, weather_path, tle_time, tle_dsma, minimize):
    update_weather_file(values, weather_path, True)
    
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
        ind = np.argmax(np.abs(tle_dsma))
        e = np.array(tle_dsma - gmat_dsma)
        return abs(e[ind])
    else:
        return np.array(tle_dsma - gmat_dsma)

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

def update_weather_file(values, path, subtract):
    assert len(values) == 90

    f10_old = values[:10]
    ap_old = [values[10 + i*8 : 10 + (i+1)*8] for i in range(10)]

    header, content = read_weather_file()

    df = pd.read_csv(StringIO(''.join(content)), sep=r'\s+', header=None, low_memory=False)
    df['date'] = df[0].astype(str) + ' ' + df[1].astype(str).str.zfill(2) + ' ' + df[2].astype(str).str.zfill(2)

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
    with open(file, 'w') as file:
        file.writelines(header)
        file.writelines(content) 

def update_ap(ap, time, error):
    for t,e in zip(time, error):
        alpha = 1
        beta = 1
        gamma = 0
        index = min(int(t * 8), 79)
        if index == 0:
            ap[index] -= round(alpha*e)
            ap[index+1] -= round(alpha*beta*e)
            ap[index+2] -= round(alpha*beta*gamma*e)
        elif index == 1:
            ap[index-1] -= round(alpha*beta*e)
            ap[index] -= round(alpha*e)
            ap[index+1] -= round(alpha*beta*e)
            ap[index+2] -= round(alpha*beta*gamma*e)
        elif index == 78:
            ap[index-2] -= round(alpha*beta*gamma*e)
            ap[index-1] -= round(alpha*beta*e)
            ap[index] -= round(alpha*e)
            ap[index+1] -= round(alpha*e)
        elif index == 79:
            ap[index-2] -= round(alpha*beta*gamma*e)
            ap[index-1] -= round(alpha*beta*e)
            ap[index] -= round(alpha*e)
        else:
            ap[index-2] -= round(alpha*beta*gamma*e)
            ap[index-1] -= round(alpha*beta*e)
            ap[index] -= round(alpha*e)
            ap[index+1] -= round(alpha*beta*e)
            ap[index+2] -= round(alpha*beta*gamma*e)
    return np.clip(ap, 0, 400)

def update_f10(f10, error):
    f10 -= np.median(error)
    return np.clip(f10, 60.0, 400.0)

def read_weather_file():
    with open("../GMAT/data/atmosphere/earth/SpaceWeather-v1.2.txt") as file:
        lines = file.readlines()
    
    header_end = next(i for i, line in enumerate(lines) if 'BEGIN OBSERVED' in line)
    header = lines[0:header_end+1]
    content = lines[header_end + 1:]
    return header, content

def plot_ap_f10(vals, med_vals):
    vals = np.array(vals)        
    med_vals = np.array(med_vals) 

    f10 = vals[:, :10]
    ap_flat = vals[:, 10:]
    f10_med = med_vals[:10]
    ap_med_flat = med_vals[10:]

    n_spacecraft, n_days = f10.shape
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

    for i in range(n_spacecraft):
        x_step = np.repeat(np.arange(n_days), 2)[1:]
        x_step = np.append(x_step, n_days)
        y = np.repeat(f10[i], 2)
        ax1.step(x_step, y, where='post', color='black', alpha=0.3)

    x_step = np.repeat(np.arange(len(f10_med)), 2)[1:]
    x_step = np.append(x_step, len(f10_med))
    y_med = np.repeat(f10_med, 2)
    ax1.step(x_step, y_med, where='post', color='red', alpha=1.0, label='Median F10.7')

    ax1.set_ylabel("F10.7")
    ax1.grid(True)
    ax1.legend()

    ap_len = ap_flat.shape[1]
    for i in range(n_spacecraft):
        x_ap = np.repeat(np.arange(ap_len) / 8, 2)[1:]
        x_ap = np.append(x_ap, ap_len / 8)
        y_ap = np.repeat(ap_flat[i], 2)
        ax2.step(x_ap, y_ap, where='post', color='black', alpha=0.3)

    x_ap = np.repeat(np.arange(len(ap_med_flat)) / 8, 2)[1:]
    x_ap = np.append(x_ap, len(ap_med_flat) / 8)
    y_ap = np.repeat(ap_med_flat, 2)
    ax2.step(x_ap, y_ap, where='post', color='red', alpha=1.0, label='Median ap index')

    ax2.set_xlabel("Days")
    ax2.set_ylabel("ap index")
    ax2.grid(True)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(f'../STORMS/{STORM}/Results/fopt.png')

def find_bias(args):
    try:
        # Sorting TLE_DATA
        tle_data, sat_id, mass = args
        epochs = np.array([x['EPOCH'] for x in tle_data], dtype='datetime64')
        ind = np.argsort(epochs)
        sorted_tle_data = [tle_data[i] for i in ind]
        sorted_epochs = epochs[ind]
        _, unique_indices = np.unique(sorted_epochs, return_index=True)
        sorted_tle_data = [sorted_tle_data[i] for i in unique_indices]

        tle_sma, tle_time, filtered_sorted_tle_data = [], [], []
        for i in sorted_tle_data:
            epoch = np.datetime64(i['EPOCH'])
            if (START_DATE <= epoch <= END_DATE):
                tle_sma.append(float(i['SEMIMAJOR_AXIS']))
                filtered_sorted_tle_data.append(i)
        
        start = np.datetime64(filtered_sorted_tle_data[0]['EPOCH'])
        for i in filtered_sorted_tle_data:
            epoch = np.datetime64(i['EPOCH'])
            tle_time.append((epoch - start) / np.timedelta64(1, 'D'))

        if len(filtered_sorted_tle_data) < 2:
            return False, False

        tle_dsma = np.gradient(tle_sma, tle_time) * 365 #km/year

        # Prepare TEMP GMAT SCRIPT and Directory
        path = '/home/hennyc/temp'
        os.makedirs(path, exist_ok=True)
        temp_dir = tempfile.mkdtemp(prefix=f'gmat_{sat_id}_', dir=path)
        script_path = os.path.join(temp_dir, f'gmat_sat_{sat_id}.script')
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
            'WEATHERFILE_VAL': WEATHER_OUT,
            'FILENAME_VAL': output_path,
            'LENGTH_VAL': round(((np.datetime64(filtered_sorted_tle_data[-1]['EPOCH']) - np.datetime64
                                (filtered_sorted_tle_data[0]['EPOCH'])) / np.timedelta64(1, 'D')),8)
        }

        for key, val in values.items():
            script = script.replace(key, str(val))
            
        with open(script_path, 'w') as f:
            f.write(script)

        # Solve for bias
        error = find_bias_error(script_path, output_path, tle_time, tle_dsma)
        if isinstance(error, bool):
            return False, False
        return tle_time, error
    
    except Exception as e:
        shutil.rmtree(temp_dir)
        return False, False

def find_bias_error(script_path, output_path, tle_time, tle_dsma):
    try:
        gmat.LoadScript(script_path)
        status = gmat.Execute()
        if status != 1:
            return False
    except Exception as e:
        return False
    finally:
        gmat.Clear()

    try:
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
        return np.array(gmat_dsma-tle_dsma)/np.array(tle_dsma) * 100
    except Exception as e:
        return False


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python fopt.py <month_year>")
        sys.exit(1)

    STORM = sys.argv[1]
    with open(f"../STORMS/{STORM}/DATES.txt", "r") as file:
        _, _, _ = file.readline().strip().split(",")
        _, START_STR_2, END_STRING_2 = file.readline().strip().split(",")
        
    START_DATE, END_DATE = np.datetime64(START_STR_2), np.datetime64(END_STRING_2)
    base = f"../STORMS/{STORM}"
    TLE_FILE = f"{base}/TLE_DATA.json"
    MASS_FILE = f"{base}/MOPT_OUTPUT.txt"
    WEATHER_OUT = f"{base}/WEATHER.txt"
    GMAT_FILE = "fopt.script"

    data = read_tle_file()

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
        vals = list(tqdm(pool.imap(find_changes, args_list), total=len(args_list), desc="FOPT"))
    
    clean_vals = [v for v in vals if not isinstance(v, bool) and v is not False]
    print(str(len(clean_vals)) + "/" + str(len(vals)) + " converged")
    med_vals = np.median(clean_vals, axis=0)

    h, c = read_weather_file()
    write_weather_file(WEATHER_OUT, h, c)
    update_weather_file(med_vals, WEATHER_OUT, False)

    plot_ap_f10(clean_vals, med_vals)


    # Find any bias
    with Pool(cpu_count()) as pool:
        bias = list(tqdm(pool.imap(find_bias, args_list), total=len(args_list), desc="Bias"))

    plt.figure(figsize=(10, 6))

    for _, (tle_time, error) in enumerate(bias):
        if isinstance(tle_time, bool) or isinstance(error, bool):
            continue
        plt.plot(tle_time, error, color='black')

    plt.xlabel('Time')
    plt.ylabel('Error (%)')
    plt.ylim([-200, 200])
    plt.title('Errors after median f10 and ap adjustments for all sats')
    plt.tight_layout()
    plt.savefig(f'../STORMS/{STORM}/Results/bias.png')
