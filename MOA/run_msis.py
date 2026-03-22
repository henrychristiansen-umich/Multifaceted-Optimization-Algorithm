"""
MSIS Runner for MOA-2
----------------------

Author: Henry Christiansen
Date:   2026-01-26
Email:  hennyc@umich.edu

Description:
    This script runs MSIS along Swarm A or B orbital tracks 
    and outputs the orbit averaged density so that MOA-2 
    can be validated.


Usage:
    python msis.py <Storm> <Swarm> <NRLMSIS Model>
"""

# imports
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates 
import pymsis 
import sys


def read_weather(filepath):
    """
    Docstring for read_weather: reads weather file and
    outputs relevant data as a dict
    
    :param filepath: system filepath to weather data file
    :type filepath: string

    :return weather_data: weather data from file as a dict
    """
    weather_data = {}
    with open(filepath, 'r') as f:
        for line in f:
            # Skip non data lines
            if line.startswith("#") or "BEGIN" in line or "NUM" in line or not line.strip():
                continue

            # should be 33 parts in each line (ap, kp, etc...)
            parts = line.strip().split()
            if len(parts) < 33:
                continue

            # assign date as key
            year = int(parts[0])
            month = int(parts[1])
            day = int(parts[2])
            date = np.datetime64(f"{year:04d}-{month:02d}-{day:02d}")

            avg_ap = int(parts[22])
            ap = list(map(int, parts[14:22]))
            f107 = float(parts[30])
            f107a = float(parts[31])

            # create dict
            weather_data[date] = {
                "avg_ap": avg_ap,
                "ap": ap,
                "f107": f107,
                "f107a": f107a,
            }

    return weather_data

def read_swarm(filename):
    """
    Docstring for read_swarm: Reads swarm data file and 
    outputs data as a dict 
    
    :param filename: system filepath to swarm .json data file
    :type filename: string

    :return swarm_data: swarm data in a dict
    """
    swarm_data = {}
    with open(filename, 'r') as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            
            # should be 9 parts in each line
            parts = line.strip().split()
            if len(parts) < 9:
                continue

            date_str = parts[0]
            time_str = parts[1]
            date = np.datetime64(f"{date_str}T{time_str}")
            alt = float(parts[3]) 
            long = float(parts[4]) 
            lat = float(parts[5]) 
            arg_lat = float(parts[7])
            density = float(parts[8])

            swarm_data[date] = {
                "alt": alt,
                "long": long,
                "lat": lat,
                "argOfLat": arg_lat,
                "density": density
            }

    return swarm_data

def convert_ap(time, ap_data):
    """
    Docstring for convert_ap: converts chronological aps 
    from weather file into the ap inputs of the MSIS models
    
    :param time: Description
    :type time: npdatetime64
    :param ap_data: Description
    :type ap_data: dict

    :return aps: ap values
    :rtype aps: array
    """
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

def get_version(str):
    """
    Docstring for get_version: returns integer version 
    for pymsis given desired msis model
    
    :param str: NRLMSIS Model String
    """
    versions = {
        "MSIS00": 0,
        "MSIS20": 2.0,
        "MSIS21": 2.1
    }
    return versions.get(str, None)

def run_msis(weather_file, swarm_data, version_str, start_date, end_date):
    """
    Docstring for run_msis: runs the NRLMSIS model along
    swarm for the given time period. Outputs the densities
    to be used for MOA-2 validation.
    
    :param weather_file: weather file system path
    :param swarm_data: swarm data dict
    :param version_str: which MSIS version to run
    :param start_date: start date of storm period
    :param end_date: end date of storm period
    """
    weather_data = read_weather(weather_file)
    dates = np.arange(start_date, end_date, np.timedelta64(30, "s"))
    densities = np.full(len(dates), np.nan)

    for i, time in enumerate(dates):
        day = time.astype("datetime64[D]")

        if day not in weather_data or time not in swarm_data:
            continue

        try: 
            f107 = weather_data[day]["f107"]
            f107a = weather_data[day]["f107a"]
            aps = convert_ap(time, weather_data)

            out = pymsis.calculate(
                time,
                swarm_data[time]["long"],
                swarm_data[time]["lat"],
                swarm_data[time]["alt"] / 1000,
                f107,
                f107a,
                aps,
                version=get_version(version_str),
                geomagnetic_activity=-1,
            )
            out = np.squeeze(out)
            densities[i] = out[pymsis.Variable.MASS_DENSITY] * 1e9
        except KeyError:
            continue

    return densities

def compute_orbit_averages(arg_lat, densities, times, wrap_threshold=-300):
    """
    Docstring for compute_orbit_averages: Detects orbit boundaries
    using argument of latitude wand computes orbit averaged
    densities, excluding the first and last incomplete orbits.
    
    :param arg_lat: argument of latitude array
    :param densities: density array
    :param times: time array
    :param wrap_threshold: change in argument of latitude threshold
    """

    arg_lat = np.asarray(arg_lat)
    densities = np.asarray(densities)

    # ---- 1. Detect orbit boundaries ----
    wrap_indices = np.where(np.diff(arg_lat) < wrap_threshold)[0]

    if len(wrap_indices) < 1:
        raise ValueError("Not enough orbit wraps detected for averaging.")

    # Orbit boundaries: each wrap index marks end of an orbit
    boundaries = [0] + (wrap_indices + 1).tolist() + [len(times)]

    avg_times = []
    orbit_avg = []

    # ---- 2. Skip first & last orbits ----
    # Valid orbit indices: 1 .. len(boundaries)-3 (inclusive)
    for i in range(1, len(boundaries) - 2):
        start = boundaries[i]
        end   = boundaries[i + 1]

        # Orbit average
        orbit_avg.append(np.nanmean(densities[start:end]))

        # Orbit midpoint timestamp
        mid = start + (end - start) // 2
        avg_times.append(times[mid])

    return np.array(avg_times), np.array(orbit_avg), boundaries

def delta_p(avg, swarm_avg):
    """
    Docstring for delta_p: calculates the delta-p of MOA-2 
    and MSIScompared to Swarm.
    
    :param avg: average density array of MOA-2 or MSIS
    :param swarm_avg: average density array of Swarm
    """
    dp = abs(np.max(avg) - np.max(swarm_avg))/(np.max(avg) + np.max(swarm_avg)/2) * 100
    return dp


def rho_t(avg, avg_times):
    """
    Docstring for rho_t: calculates the rho-t of MOA-2
    and MSIS compared to Swarm.
    
    :param avg: average density array of MOA-2 or MSIS
    :param avg_times: average time array (x-axis for density array)
    """
    times_sec = (avg_times - avg_times[0]).astype('timedelta64[s]').astype(float)
    rho_t = np.trapezoid(avg, x=times_sec)
    return rho_t

def plot_and_save(storm_str, swarm_str, version, avg_times, moa_avg, msis_avg, swarm_avg, start_date, end_date):
    """
    Docstring for plot_and_save: plots results and saves
    statistics to results file.
    
    :param storm_str: string of storm (ex: OCT_2024)
    :param swarm_str: A or B (Swarm A or Swarm B)
    :param version: MSIS version looked at (MSIS-00, MSIS 2.0 etc...)
    :param avg_times: array holding the times for each density array
    :param moa_avg: density array of orbit averaged MOA-2 densities
    :param msis_avg: density array of orbit averaged MSIS densities
    :param swarm_avg: density array of orbit averaged Swarm (A or B) densities
    :param start_date: start date of storm period
    :param end_date: end date of strom period
    """
    fig = plt.figure(figsize=(9, 6))
    ax_ap = fig.add_axes([0.363, 0.1, 0.57, 0.8])
    size = 15

    bbox = ax_ap.get_position()
    x_center = bbox.x0 + bbox.width / 2

    month_dict = {"MAR": "03", "APR": "04", "MAY": "05", "AUG": "08", "SEP": "09", "OCT": "10"}
    m,y = storm_str.split('_')
    storm_fig_str = f"{month_dict.get(m, None)}-{y}"
    fig.text(x_center, 0.98, f"{storm_fig_str} SWARM-{swarm_str}: Orbit-averaged Densities",
        ha='center', va='top', fontsize=size, fontweight='bold')
    
    versions = {0: "NRLMSIS-00", 2.0: "NRLMSIS 2.0", 2.1: "NRLMSIS 2.1"}
    ax_ap.plot(avg_times, msis_avg * (10**5) , linewidth=3, color='blue', label=versions.get(version, None))
    ax_ap.plot(avg_times, moa_avg * (10**5) , color='red', linewidth=3, label='MOA-2')
    ax_ap.plot(avg_times, swarm_avg * (10**5) , color='black', linewidth=3, label=f'Swarm {swarm_str}')
    ax_ap.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax_ap.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    ax_ap.tick_params(labelsize=size)
    ax_ap.set_xlabel("Real Time", fontweight='bold', fontsize=size)
    ax_ap.set_ylabel(r"Density ($\mathbf{10^{-5}}$ kg/km$\mathbf{^3}$)", fontweight='bold', fontsize=size)
    ax_ap.set_xlim(start_date, end_date)
    ax_ap.legend(loc='upper left', frameon=False, fontsize=size)
    fig.savefig(f'/home/hennyc/data/{storm_str}/Results/swarm{swarm_str}.png')
    plt.close(fig)

    msis_dp = delta_p(msis_avg, swarm_avg)
    moa_dp = delta_p(moa_avg, swarm_avg)
    
    msis_rmse = np.sqrt(np.mean((np.array(swarm_avg) - np.array(msis_avg)) ** 2)) * 10**5
    moa_rmse = np.sqrt(np.mean((np.array(swarm_avg) - np.array(moa_avg)) ** 2)) * 10**5

    msis_rhot = rho_t(msis_avg, avg_times)
    moa_rhot = rho_t(moa_avg, avg_times)
    swarm_rhot = rho_t(swarm_avg, avg_times)

    with open(f'/home/hennyc/data/{storm_str}/Results/swarm{swarm_str}_results.txt', "w") as f:
        f.write("------------\n")
        f.write(f"{versions.get(version, None)} Dp         : " + str(round(msis_dp, 2)) + "\n")
        f.write("MOA-2 Dp        : " + str(round(moa_dp, 2)) + "\n")
        f.write("Reduction in Dp : " + str(round((msis_dp - moa_dp) / msis_dp * 100, 2)) + "\n")
        f.write("--\n")
        f.write(f"{versions.get(version, None)} RMS Error: " + str(round(msis_rmse, 2)) + "\n")
        f.write("MOA-2 RMS Error : " + str(round(moa_rmse, 2)) + "\n")
        f.write("Reduction       : " + str(round((msis_rmse - moa_rmse) / msis_rmse * 100, 2)) + "\n")
        f.write("--\n")
        f.write(f"{versions.get(version, None)} rhot       : " + str(round(msis_rhot, 2)) + "\n")
        f.write("MOA-2 rhot        : " + str(round(moa_rhot, 2)) + "\n")
        f.write("SWARM rhot        : " + str(round(swarm_rhot, 2)) + "\n")
        f.write("Reduction         : " + str(round((abs(swarm_rhot - msis_rhot) - abs(swarm_rhot - moa_rhot)) / abs(swarm_rhot - msis_rhot) * 100, 2)) + "\n")
        f.write("------------\n")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python msis.py <Storm> <Swarm> <NRLMSIS Model>")
        sys.exit(1)
    
    storm = sys.argv[1].upper()
    swarm = sys.argv[2].upper()
    model = sys.argv[3].upper()

    if swarm not in ("A", "B"):
        print("<Swarm> must be 'A' or 'B'")
        print("Usage: python msis.py <Storm> <Swarm> <NRLMSIS Model>")
        sys.exit(1)
    
    if model not in ("MSIS00", "MSIS20", "MSIS21"):
        print("<NRLMSIS Model> must be MSIS00, MSIS20, MSIS21")
        print("Usage: python msis.py <Storm> <Swarm> <NRLMSIS Model>")
        sys.exit(1)


    base_path = "/home/hennyc"
    date_file = f"{base_path}/data/{storm}/DATES.txt"
    with open(date_file, "r") as file:
        _, _, _ = file.readline().strip().split(",")
        _, start_str, end_str = file.readline().strip().split(",")

    start_date = np.datetime64(start_str)
    end_date = np.datetime64(end_str)

    msis_file = f"{base_path}/gmat-git/GMAT/data/atmosphere/earth/SpaceWeather-All-v1.2.txt"
    moa_file = f"{base_path}/data/{storm}/WEATHER.txt"
    swarm_file = f"{base_path}/data/{storm}/DENSITY_DATA/SWARM{swarm}.txt"

    # Start Comparison
    times = np.arange(start_date, end_date, np.timedelta64(30, "s"))
    swarm_data = read_swarm(swarm_file)
    swarm_density = []
    arg_lat = []
    for t in times:
        if t in swarm_data:
            swarm_density.append(swarm_data[t]["density"] * 1e9)
            arg_lat.append(swarm_data[t]["argOfLat"])
        else:
            swarm_density.append(np.nan)
            arg_lat.append(np.nan)

    swarm_density = np.array(swarm_density)
    arg_lat = np.array(arg_lat)

    # Run MSIS
    moa_density = run_msis(moa_file, swarm_data, model, start_date, end_date)
    msis_density = run_msis(msis_file, swarm_data, model, start_date, end_date)

    avg_times, swarm_avg, _ = compute_orbit_averages(arg_lat, swarm_density, times)
    _, moa_avg, _ = compute_orbit_averages(arg_lat, moa_density, times)
    _, msis_avg, _ = compute_orbit_averages(arg_lat, msis_density, times)

    plot_and_save(storm, swarm, model, avg_times, moa_avg, msis_avg, swarm_avg, 
                  start_date, end_date)