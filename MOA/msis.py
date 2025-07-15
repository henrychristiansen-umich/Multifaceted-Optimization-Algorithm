import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates 
from datetime import datetime
import pymsis 

def Read_Weather_Data(filepath):
    weather_data = {}
    with open(filepath, 'r') as f:
        for line in f:
            if line.startswith("#") or "BEGIN" in line or "NUM" in line or not line.strip():
                continue

            parts = line.strip().split()
            if len(parts) < 33:
                continue

            year = int(parts[0])
            month = int(parts[1])
            day = int(parts[2])
            date = np.datetime64(f"{year:04d}-{month:02d}-{day:02d}")

            avg_ap = int(parts[22])
            ap = list(map(int, parts[14:22]))
            f107 = float(parts[30])
            f107a = float(parts[31])

            weather_data[date] = {
                "avg_ap": avg_ap,
                "ap": ap,
                "f107": f107,
                "f107a": f107a,
            }
    return weather_data

def Read_SWARM_Data(filename):
    swarm_data = {}
    with open(filename, 'r') as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue

            parts = line.strip().split()
            if len(parts) < 9:
                continue

            date_str = parts[0]
            time_str = parts[1]
            timestamp = np.datetime64(f"{date_str}T{time_str}")
            altitude = float(parts[3]) 
            longitude = float(parts[4]) 
            latitude = float(parts[5]) 
            arg_lat = float(parts[7])
            density = float(parts[8])

            swarm_data[timestamp] = {
                "alt": altitude,
                "long": longitude,
                "lat": latitude,
                "argOfLat": arg_lat,
                "density": density
            }

    return swarm_data

def Convert_ap(time, ap_data):
    hour = int(str(time)[11:13])
    minute = int(str(time)[14:16])
    second = int(str(time)[17:19])
    total_seconds = hour * 3600 + minute * 60 + second
    ap_index = total_seconds // (3 * 3600)

    today = time.astype("datetime64[D]")
    get_ap = lambda d: ap_data.get(d, {"ap": [0]*8})["ap"]
    ap_today = ap_data.get(today)

    ap0 = ap_today["avg_ap"]
    ap1 = ap_today["ap"][ap_index]
    ap2 = get_ap(today)[ap_index - 1] if ap_index >= 1 else get_ap(today - 1)[ap_index - 1 + 8]
    ap3 = get_ap(today)[ap_index - 2] if ap_index >= 2 else get_ap(today - 1)[ap_index - 2 + 8]
    ap4 = get_ap(today)[ap_index - 3] if ap_index >= 3 else get_ap(today - 1)[ap_index - 3 + 8]
    ap5_vals = get_ap(today - 1)[4:] + get_ap(today - 2)[:4]
    ap6_vals = get_ap(today - 2)[4:] + get_ap(today - 3)[:4]
    ap5 = np.mean(ap5_vals)
    ap6 = np.mean(ap6_vals)

    return [[ap0, ap1, ap2, ap3, ap4, ap5, ap6]]

def Run_MSIS(filename, swarm, start_date, end_date):
    data = Read_Weather_Data(filename)
    dates = np.arange(start_date, end_date, np.timedelta64(30, "s"))
    densities = []

    for time in dates:
        day = time.astype("datetime64[D]")
        f107 = data[day]["f107"]
        f107a = data[day]["f107a"]
        aps = Convert_ap(time, data)

        out = pymsis.calculate(time, swarm[time]["long"], swarm[time]["lat"], swarm[time]["alt"]/1000, f107, f107a, aps, version=0, geomagnetic_activity=-1)
        out = np.squeeze(out)
        densities.append(out[pymsis.Variable.MASS_DENSITY] * 1e9)

    return np.array(densities)

def Run_Comparison(uncorrected, corrected, swarm, start_date, end_date):
    times = np.arange(start_date, end_date, np.timedelta64(30, "s"))
    s = 'SWARM A'
    #year = swarm_a[36:40]
    #month = swarm_a[41:43]
    year = 2017
    month = 5
    swarm = Read_SWARM_Data(swarm)
    swarm_density = []
    arg_lat = []
    for time in times:
        swarm_density.append(swarm[time]["density"] * 1e9)
        arg_lat.append(swarm[time]["argOfLat"])
    
    swarm_density = np.array(swarm_density)
    arg_lat = np.array(arg_lat)
    moa_density = Run_MSIS(corrected, swarm, start_date, end_date)
    uncorrected_density = Run_MSIS(uncorrected, swarm, start_date, end_date)

    swarm_avg = []
    moa_avg = []
    uncorrected_avg = []
    avg_times = []
    start = np.floor(swarm[times[0]]["argOfLat"])
    period = 0

    for i, time in enumerate(times[1:]):
        if np.floor(swarm[time]["argOfLat"]) == start or np.round(swarm[time]["argOfLat"]) == start:
            period = i + 1
            break
        
    for start in np.arange(0, len(times), period):
        end = min(start + period, len(times))
        if end == len(times):
            break
        avg_times.append(times[end])
        swarm_avg.append(np.mean(swarm_density[start:end]))
        moa_avg.append(np.mean(moa_density[start:end]))
        uncorrected_avg.append(np.mean(uncorrected_density[start:end]))
    
    avg_times = np.array(avg_times)
    swarm_avg = np.array(swarm_avg)
    moa_avg = np.array(moa_avg)
    uncorrected_avg = np.array(uncorrected_avg)

    plt.figure(figsize=(7, 5))
    plt.plot(avg_times, uncorrected_avg, color='red', label="NRLMSISE-00")
    plt.plot(avg_times, moa_avg, color='blue', label='MOA')
    plt.plot(avg_times, swarm_avg, color='black', label='Swarm Accelerometer')
    plt.fill_between(avg_times, swarm_avg - (swarm_avg * 0.2), (swarm_avg * 0.2) + swarm_avg , color='black', 
                     alpha=0.3, edgecolor='none', label='\u00b1 20% Swarm Variation')
    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    plt.xlabel("Real Time")
    plt.ylabel("Density (kg/km³)")
    plt.title(f"{s}: Orbit-averaged Nuetral Densities {month}-{year}")
    plt.xlim(start_date, end_date)
    plt.tight_layout()
    plt.legend()
    plt.show()

    print("-------")
    print("MSIS Dp: " + str(Delta_P(uncorrected_avg, swarm_avg)))
    print("MOA Dp: " + str(Delta_P(moa_avg, swarm_avg)))
    print("Swarm eta: " + str(Eta(swarm_avg, avg_times)))
    print("MSIS eta: " + str(Eta(uncorrected_avg, avg_times)))
    print("MOA eta: " + str(Eta(moa_avg, avg_times)))
    print("MSIS time: " + str(Time_l(uncorrected_avg, swarm_avg, avg_times)))
    print("MOA time: " + str(Time_l(moa_avg, swarm_avg, avg_times)))
    print("SWARM rhot: " + str(Rho_T(swarm_avg, swarm_avg, avg_times)))
    print("MSIS rhot: " + str(Rho_T(uncorrected_avg, swarm_avg, avg_times)))
    print("MOA rhot: " + str(Rho_T(moa_avg, swarm_avg, avg_times)))
    print("-------")
    
def Delta_P(avg, swarm_avg):
    dp = abs(np.max(avg) - np.max(swarm_avg))/(np.max(avg) + np.max(swarm_avg)/2) * 100
    return dp

def Eta(avg, avg_times):
    peak_idx = np.argmax(avg)
    peak_time = avg_times[peak_idx]
    lower_bound_time = peak_time - np.timedelta64(24, 'h')
    mask = (avg_times < peak_time) & (avg_times >= lower_bound_time)
    preceding_values = avg[mask]
    return avg[peak_idx] / np.mean(preceding_values)

def Time_l(avg, swarm_avg, avg_times):
    idx_avg = np.argmax(avg)
    idx_swarm = np.argmax(swarm_avg)
    return (avg_times[idx_avg] - avg_times[idx_swarm]).astype('timedelta64[s]').astype(float) / 3600

def Rho_T(avg, swarm_avg, avg_times):
    window = np.timedelta64(12, 'h')
    lower_idx = None
    upper_idx = None
    for i, t in enumerate(avg_times):
        prev_mask = (avg_times >= t - window) & (avg_times < t)
        if np.sum(prev_mask) < 2:
            continue
        mean = np.mean(swarm_avg[prev_mask])
        std = np.std(swarm_avg[prev_mask])
        next_mask = (avg_times >= t) & (avg_times < t + window)
        if np.all(swarm_avg[next_mask] > mean + std):
            lower_idx = i
            break
    
    for i in reversed(range(len(avg_times))):
        t = avg_times[i]
        next_mask = (avg_times > t) & (avg_times <= t + window)
        if np.sum(next_mask) < 2:
            continue
        mean = np.mean(swarm_avg[next_mask])
        std = np.std(swarm_avg[next_mask])
        prev_mask = (avg_times > t - window) & (avg_times <= t)
        if np.all(swarm_avg[prev_mask] > mean + std):
            upper_idx = i
            break

    times_sec = (avg_times[lower_idx:upper_idx + 1] - avg_times[0]).astype('timedelta64[s]').astype(float)
    return np.trapezoid(avg[lower_idx:upper_idx + 1], x=times_sec)


STORM = "APR_2023"

DATE_FILE = f"../STORMS/{STORM}/DATES.txt"
with open(DATE_FILE, "r") as file:
    _ = file.readline()
    _, start_str, end_str = file.readline().strip().split(",")

START_DATE = np.datetime64(start_str)
END_DATE = np.datetime64(end_str)

origninal_cssi = "../GMAT_R2025a/data/atmosphere/earth/SpaceWeather-All-v1.2.txt"
fopt_output = "../STORMS/" + STORM + "/WEATHER.txt"
champ = "../STORMS/" + STORM + "/DENSITY_DATA/SWARMB.txt"
Run_Comparison(origninal_cssi, fopt_output, champ, START_DATE, END_DATE)
