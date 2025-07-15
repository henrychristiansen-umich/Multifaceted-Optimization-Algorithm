# F10.7 and ap adjustments solver for MOA
#
# Coded by Henry Christiansen. University of Michigan.
#
# This file solves for the F10.7 and 3 hour ap adjustments 
# that minimize the rms difference between GMAT and TLE data

# IMPORTS
from scipy.ndimage import gaussian_filter1d
from multiprocessing import Pool, cpu_count
import traceback
from scipy.optimize import minimize_scalar
from collections import defaultdict
from scipy.stats import linregress
import matplotlib.pyplot as plt
from datetime import timedelta
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

STORM = "APR_2023"
MAX_ITER = 25

with open(f"../STORMS/{STORM}/DATES.txt", "r") as file:
    _, _, _ = file.readline().strip().split(",")
    _, START_STR_2, END_STRING_2 = file.readline().strip().split(",")
    
START_DATE, END_DATE = np.datetime64(START_STR_2), np.datetime64(END_STRING_2)
base = f"../STORMS/{STORM}"
TLE_FILE = f"{base}/TLE_DATA.json"
MASS_FILE = f"{base}/MOPT_OUTPUT.txt"
WEATHER_OUT = f"{base}/WEATHER.txt"
WEATHER_IN = "../GMAT_R2025a/data/atmosphere/earth/SpaceWeather-All-v1.2.txt"
GMAT_FILE = "newfopt.script"
SMA_FILE = "../GMAT_OUTPUT/PROP_SMA.txt"
TIME_FILE = "../GMAT_OUTPUT/PROP_TIME.txt"

def read_tle_file():
    with open(TLE_FILE, 'r') as f:
        tle_data = json.load(f)
    data = defaultdict(list)
    for i in tle_data:
        name = i["NORAD_CAT_ID"]
        data[name].append(i)

    return data

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

        # make log files
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, f"sat_{sat_id}.log")
        with open(log_path, 'w') as log:
                log.write(f"Starting values solve for {sat_id}\n")

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
        temp_dir = tempfile.mkdtemp(prefix=f'gmat_{sat_id}_')
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

        # Solve for changes
        error = None
        last_error = None
        for _ in range(MAX_ITER):
            error = find_error(np.concatenate((f10, ap)), script_path, output_path, weather_path, tle_time, tle_dsma, log_path, False)
            if error is np.nan:
                return False

            if last_error is not None:
                if abs(np.sqrt(np.mean(last_error**2)) - np.sqrt(np.mean(error**2))) < 1:
                    # shutil.rmtree(temp_dir)
                    # return np.concatenate((f10, ap))
                    break

            f10 = update_f10(f10, tle_time, error)
            ap = update_ap(ap, tle_time, error)
            last_error = error
        
        with open(log_path, 'a') as log:
            log.write(f"\nSTARTING MINIMIZE !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")
        
        day = int(np.floor(tle_time[np.argmax(np.abs(tle_dsma))])) #+ 1
        f10_base = f10.copy()

        result = minimize_scalar(
            lambda val: find_error(
                np.concatenate((
                    np.array([
                        val if i == day else f10_base[i]
                        for i in range(len(f10_base))
                    ]),
                    ap
                )),
                script_path, output_path, weather_path, tle_time, tle_dsma, log_path, True
            ),
            bounds=(60.0, 400.0),
            method='bounded',
            options={'xatol': 1, 'maxiter': 50}
        )

        shutil.rmtree(temp_dir)
        if result.success:
            f10[day] = result.x
            with open(log_path, 'a') as log:
                log.write(f"\n CONVERGED !!!!!!!!!!!!!!!!!!!!!!!!!!\n")
                log.write(f"f10: {f10}")
            return np.concatenate((f10, ap))
        else:
            with open(log_path, 'a') as log:
                log.write(f"\n NO CONVERGE !!!!!!!!!!!!!!!!!!!!!!!!!!\n")
                log.write(f"f10: {f10}")
            return np.concatenate((f10, ap))
    except Exception as e:
        with open(log_path, 'a') as log:
            log.write(f"Exception: {e}")
        shutil.rmtree(temp_dir)
        return False
  
def find_error(values, script_path, output_path, weather_path, tle_time, tle_dsma, log_path, minimize):
    with open(log_path, 'a') as log:
        log.write(f"Iteration: Vals = {values}\n")

    update_weather_file(values, weather_path, True)
    
    try:
        gmat.LoadScript(script_path)
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

    # Load CSV for easier date lookup
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
        beta = 0
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

def update_f10(f10, time, error):
    i = np.floor(time).astype(int)
    for start in np.arange(0, 10, 1):
        alpha = 1
        indeces = (i >= start) & (i < start+1)
        if np.any(indeces):
            selected_errors = error[indeces]
            if len(selected_errors) == 1:
                avg = selected_errors[0]
            else:
                avg = np.median(selected_errors)
            f10[start] -= avg * alpha
    f10 = np.round(f10, 1)
    return np.clip(f10, 60.0, 400.0)

def read_weather_file():
    with open(WEATHER_IN) as file:
        lines = file.readlines()
    
    header_end = next(i for i, line in enumerate(lines) if 'BEGIN OBSERVED' in line)
    header = lines[0:header_end+1]
    content = lines[header_end + 1:]
    return header, content

if __name__ == '__main__':
    data = read_tle_file()
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
