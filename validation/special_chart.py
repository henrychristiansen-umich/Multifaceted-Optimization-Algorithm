
import argparse
from matplotlib.patches import Patch
import matplotlib.colors as mcolors
from matplotlib.ticker import FixedLocator
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker 
from datetime import datetime
from io import StringIO
from tqdm import tqdm
import pandas as pd
import numpy as np
import pymsis

def plot_all(storm, vals, avg_timesa, avg_timesb, moa_avga, moa_avgb, moa_uppera, moa_upperb, moa_lowera, moa_lowerb, msis_avga, msis_avgb, swarm_avga, swarm_avgb, start_date, end_date):
    size = 14
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
    ap_bins = np.arange(0, 400 + ap_bin_width, ap_bin_width)  # y-axis bins
    N_ap_bins = len(ap_bins) - 1

    ap_count_matrix = np.zeros((N_ap_bins, N_ap))

    for col_idx in range(N_ap):
        col_values = ap_vals[:, col_idx]
        counts, _ = np.histogram(col_values, bins=ap_bins)
        ap_count_matrix[:, col_idx] = counts

    norm_ap = mcolors.PowerNorm(
        gamma=1,
        vmin=0,
        vmax=int(np.round(ap_count_matrix[1:-1, :].max() / 10) * 10)
    )

    f10_vals = vals[:, :10]  
    N_spacecraft, N_f10 = f10_vals.shape
    f10_bin_width = 5
    f10_bins = np.arange(60.0, 400.0 + f10_bin_width, f10_bin_width)  # y-axis bins
    N_f10_bins = len(f10_bins) - 1

    f10_count_matrix = np.zeros((N_f10_bins, N_f10))

    for col_idx in range(N_f10):
        col_values = f10_vals[:, col_idx]
        counts, _ = np.histogram(col_values, bins=f10_bins)
        f10_count_matrix[:, col_idx] = counts

    norm_f10 = mcolors.PowerNorm(
        gamma=1,
        vmin=0,
        vmax=int(np.round(f10_count_matrix[1:-1, :].max() / 10) * 10)
    )

    fig = plt.figure(figsize=(18, 11))
    gs = fig.add_gridspec(
        2, 5,
        width_ratios=[0.03, 0.22, 1, 0.15, 1],
        wspace=0,
        hspace=0.11
    )


    cax_f10 = fig.add_subplot(gs[0, 0])
    cax_ap  = fig.add_subplot(gs[1, 0])

    ax_f10  = fig.add_subplot(gs[0, 2])
    ax_ap   = fig.add_subplot(gs[1, 2])

    ax_den1 = fig.add_subplot(gs[0, 4])
    ax_den2 = fig.add_subplot(gs[1, 4])

    sm_f10 = plt.cm.ScalarMappable(
    norm=norm_f10,
    cmap='gray_r'
    )
    sm_f10.set_array([])

    cb_f10 = fig.colorbar(sm_f10, cax=cax_f10)
    
    sm_ap = plt.cm.ScalarMappable(
    norm=norm_ap,
    cmap='gray_r'
    )
    sm_ap.set_array([])

    cb_ap = fig.colorbar(sm_ap, cax=cax_ap)

    # for cb in [cb_f10, cb_ap]:
    #     cb.set_label(
    #         'Number of Spacecraft',
    #         fontsize=size,
    #         fontweight='bold'
    #     )
    #     cb.ax.tick_params(labelsize=size)

    for cb in [cb_f10, cb_ap]:
        ticks = cb.get_ticks()
        labels = [f"{int(t)}" for t in ticks]

        if len(labels) > 0:
            labels[-1] += "+"

        cb.locator = FixedLocator(ticks)
        cb.set_ticklabels(labels, fontsize=size)



    x_edges = np.arange(N_f10 + 1)
    y_edges = f10_bins

    pcm_f10 = ax_f10.pcolormesh(
        x_edges,
        y_edges,
        f10_count_matrix,
        cmap='gray_r',
        norm=norm_f10,
        shading='auto',
        zorder=1
    )

    ax_f10.plot(
        np.arange(len(f10_original[:N_f10])) + 0.5,
        f10_original[:N_f10],
        color="blue",
        linestyle='-',
        linewidth=3
    )

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

    f10_med_vals = med_vals[:10]

    ax_f10.plot(
        np.arange(len(f10_med_vals)) + 0.5,
        f10_med_vals,
        color="red",
        linestyle='-',
        linewidth=3
    )

    ax_f10.set_xlim(0, N_f10)

    ax_f10.set_ylim(
        np.minimum(np.min(f10_vals), np.min(f10_original[:N_f10])) - 10,
        np.maximum(np.max(f10_vals), np.max(f10_original[:N_f10])) + 50
    )

    ax_f10.tick_params(labelsize=size)

    ax_f10.set_ylabel(
        "F10.7 (sfu)",
        fontweight='bold',
        fontsize=size
    )

    ax_f10.yaxis.set_ticks_position('left')
    ax_f10.yaxis.set_label_position('left')

    ax_f10.xaxis.set_major_locator(mdates.DayLocator(interval=2))
    ax_f10.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))

    ax_f10.legend(
        handles=[legend_rect, legend_rect2],
        loc='upper left',
        frameon=False,
        fontsize=size
    )

    ax_f10.set_title(
        "(a) F10.7 Adjustment Histogram",
        fontsize=size,
        fontweight='bold'
    )

    ax_ap.set_title(
        r"(b) $\mathbf{a_p}$ Adjustment Histogram",
        fontsize=size,
        fontweight='bold'
    )

    ax_den1.set_title(
        "(c) Swarm A",
        fontsize=size,
        fontweight='bold'
    )

    ax_den2.set_title(
        "(d) Swarm B",
        fontsize=size,
        fontweight='bold'
    )

    # --- Ap heatmap ---
    x_edges = np.arange(N_ap + 1)
    y_edges = ap_bins

    pcm_ap = ax_ap.pcolormesh(
        x_edges,
        y_edges,
        ap_count_matrix,
        cmap='gray_r',
        norm=norm_ap,
        shading='auto',
        zorder=1
    )

    ax_ap.plot(
        np.arange(len(ap_original[:N_ap])) + 0.5,
        ap_original[:N_ap],
        color="blue",
        linewidth=3
    )

    legend_rect = Patch(
        facecolor="blue",
        edgecolor="none",
        label=r"Observed $a_p$"
    )

    legend_rect2 = Patch(
        facecolor="red",
        edgecolor="none",
        label=r"Median Adjusted $a_p$"
    )

    ap_med_vals = med_vals[10:]

    ax_ap.plot(
        np.arange(len(ap_med_vals)) + 0.5,
        ap_med_vals,
        color="red",
        linewidth=3
    )

    ax_ap.set_xlim(0, N_ap)
    ax_ap.set_ylim(0, 400)

    ax_ap.tick_params(labelsize=size)

    ax_ap.set_xlabel(
        "Real Time",
        fontweight='bold',
        fontsize=size
    )

    ax_ap.set_ylabel(
        r"3-hr $\mathbf{a_p}$ (nT)",
        fontweight='bold',
        fontsize=size
    )

    ax_ap.yaxis.set_ticks_position('left')
    ax_ap.yaxis.set_label_position('left')

    tick_positions = np.arange(0, N_ap + 8, 16)

    tick_labels = [
        pd.to_datetime(
            (START_DATE + np.timedelta64(2*i, 'D'))
            .astype('M8[D]')
            .astype(str)
        ).strftime('%m-%d')
        for i in range(len(tick_positions))
    ]

    ax_ap.set_xticks(tick_positions)
    ax_ap.set_xticklabels(tick_labels)

    ax_ap.legend(
        handles=[legend_rect, legend_rect2],
        loc='upper left',
        frameon=False,
        fontsize=size
    )







    ax_den1.plot(
        avg_timesa,
        swarm_avga,
        color='black',
        linewidth=3,
        label='Swarm A'
    )

    ax_den1.plot(
        avg_timesa,
        msis_avga,
        color='blue',
        linewidth=3,
        label="NRLMSISE-00"
    )

    ax_den1.plot(
        avg_timesa,
        moa_avga,
        color='red',
        linewidth=3,
        label='MOA (Median)'
    )

    ax_den1.fill_between(
        avg_timesa,
        moa_lowera,
        moa_uppera,
        color='red',
        alpha=0.25,
        label='MOA IQR'
    )

    ax_den1.tick_params(labelsize=size)

    ax_den1.set_ylabel(
        r"Density (kg/m$\mathbf{^3}$)",
        fontweight='bold',
        fontsize=size
    )

    ax_den1.set_xlim(start_date, end_date)

    ax_den1.legend(
        loc='upper left',
        frameon=False,
        fontsize=size
    )

    ax_den1.tick_params(labelbottom=False)
    
    
    ax_den2.plot(
        avg_timesb,
        swarm_avgb,
        color='black',
        linewidth=3,
        label='Swarm B'
    )

    ax_den2.plot(
        avg_timesb,
        msis_avgb,
        color='blue',
        linewidth=3,
        label="NRLMSISE-00"
    )

    ax_den2.plot(
        avg_timesb,
        moa_avgb,
        color='red',
        linewidth=3,
        label='MOA (Median)'
    )

    ax_den2.fill_between(
        avg_timesb,
        moa_lowerb,
        moa_upperb,
        color='red',
        alpha=0.25,
        label='MOA IQR'
    )

    ax_den2.xaxis.set_major_locator(mdates.DayLocator(interval=2))
    ax_den2.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))

    ax_den2.tick_params(labelsize=size)

    ax_den2.set_xlabel(
        "Real Time",
        fontweight='bold',
        fontsize=size
    )

    ax_den2.set_ylabel(
        r"Density (kg/m$\mathbf{^3}$)",
        fontweight='bold',
        fontsize=size
    )

    ax_den2.set_xlim(start_date, end_date)

    ax_den2.legend(
        loc='upper left',
        frameon=False,
        fontsize=size
    )


    ax_f10.tick_params(labelbottom=False)

    plt.savefig(f"/home/hennyc/src/total_{storm}.png", dpi=300, bbox_inches="tight")

def read_weather_file():
    """
    Docstring for read_weather_file: reads observed space weather data
    """
    with open("/home/hennyc/gmat-git_two/GMAT-R2026a-Linux-x64/data/atmosphere/earth/SpaceWeather-All-v1.2.txt") as file:
        lines = file.readlines()
    
    header_end = next(i for i, line in enumerate(lines) if 'BEGIN OBSERVED' in line)
    header = lines[0:header_end+1]
    content = lines[header_end + 1:]
    return header, content

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

def run_msis(weather_file, swarm_data, model, start_date, end_date):
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
    densities_21 = np.full(len(dates), np.nan)

    for i, time in enumerate(tqdm(dates)):
        day = time.astype("datetime64[D]")

        if day not in weather_data or time not in swarm_data:
            continue
        
        if model["version"] != 90:
            try: 
                f107 = weather_data[day]["f107"]
                f107a = weather_data[day]["f107a"]
                aps, *_ = convert_ap(time, weather_data)

                out = pymsis.calculate(
                    time,
                    swarm_data[time]["long"],
                    swarm_data[time]["lat"],
                    swarm_data[time]["alt"] / 1000,
                    f107,
                    f107a,
                    aps,
                    version=model["version"],
                    geomagnetic_activity=-1,
                )
                out21 = pymsis.calculate(
                    time,
                    swarm_data[time]["long"],
                    swarm_data[time]["lat"],
                    swarm_data[time]["alt"] / 1000,
                    f107,
                    f107a,
                    aps,
                    version=2.1,
                    geomagnetic_activity=-1,
                )
                out = np.squeeze(out)
                out21 = np.squeeze(out21)
                densities[i] = out[pymsis.Variable.MASS_DENSITY] # kg/m^3
                densities_21[i] = out21[pymsis.Variable.MASS_DENSITY] # kg/m^3
            except KeyError:
                continue
        else:

            f107 = weather_data[day]["f107"]
            f107a = weather_data[day]["f107a"]
            aps, ap0, ap1, ap2, ap3, ap4, ap5, ap6 = convert_ap(time, weather_data)

            out21 = pymsis.calculate(
                time,
                swarm_data[time]["long"],
                swarm_data[time]["lat"],
                swarm_data[time]["alt"] / 1000,
                f107,
                f107a,
                aps,
                version=2.1,
                geomagnetic_activity=-1,
            )
            out21 = np.squeeze(out21)
            densities_21[i] = out21[pymsis.Variable.MASS_DENSITY] # kg/m^3

            dtime = time.astype(datetime.datetime)
            yr = dtime.strftime("%y")
            doy = dtime.strftime("%j")
            h = dtime.strftime("%H")
            m = dtime.strftime("%M")
            s = dtime.strftime("%S")

            iyd = f"{yr}{doy}"
            sec = float(s) + 60.0*float(m) + 3600.0*float(h)
            alt = swarm_data[time]['alt']/1000
            lat = swarm_data[time]['lat']
            long = swarm_data[time]['long']
            stl = (sec/3600.0 + long/15.0) % 24.0

            base = "./msise90"
            with open(f"{base}/msis_drives.txt", "w") as f:
                f.write(
                    f"{iyd},{sec},{alt},{lat},"
                    f"{long},{stl},{f107a},{f107}"
                    f"{ap0},{ap1},{ap2},{ap3},{ap4},{ap5},{ap6}"
                )

            subprocess.run([f"{base}/msise90.exe"], check=True)

            with open(f"{base}/msis_out.txt", "r") as f:
                line = f.readline().strip()
            
            parts = line.split()
            densities[i] = float(parts[5]) # kg/m^3

    return densities, densities_21

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

    return [[ap0, ap1, ap2, ap3, ap4, ap5, ap6]], ap0, ap1, ap2, ap3, ap4, ap5, ap6


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("storm")
    parser.add_argument("model")
    args = parser.parse_args()

    storm = args.storm
    model = args.model

    model_info = {
        "MSISE90": {"version": 90, "label": "MSISE-90"},
        "NRLMSISE00": {"version": 0,  "label": "NRLMSISE-00"},
        "MSIS21": {"version": 2.1, "label": "NRLMSIS 2.1"},
    }

    model = model.strip().upper()
    if model not in model_info:
        raise ValueError(f"Invalid MSIS model: {model}")
    model = model_info[model]

    date_file = f"/home/hennyc/data/{storm}/DATES.txt"

    with open(date_file, "r") as file:
        _, _, _ = file.readline().strip().split(",")
        _, START_STR_2, END_STRING_2 = file.readline().strip().split(",")
        
    START_DATE, END_DATE = np.datetime64(START_STR_2), np.datetime64(END_STRING_2)

    BASE_PATH = f"/home/hennyc"
    clean_vals = np.load(f'{BASE_PATH}/data/{storm}/Results/fopt_adjustments.npy')


    msis_file = f"{BASE_PATH}/gmat-git/GMAT/data/atmosphere/earth/SpaceWeather-All-v1.2.txt"
    moa_file = f"{BASE_PATH}/data/{storm}/WEATHER.txt"
    moa_l_file = f"{BASE_PATH}/data/{storm}/WEATHER_MEAN_LOWER.txt"
    moa_h_file = f"{BASE_PATH}/data/{storm}/WEATHER_MEAN_UPPER.txt"
    swarma_file = f"{BASE_PATH}/data/{storm}/DENSITY_DATA/SWARMA.txt"
    swarmb_file = f"{BASE_PATH}/data/{storm}/DENSITY_DATA/SWARMB.txt"

    # Start Comparison
    times = np.arange(START_DATE, END_DATE, np.timedelta64(30, "s"))
    swarma_data = read_swarm(swarma_file)
    swarm_densitya= []
    arg_lata = []
    for t in times:
        if t in swarma_data:
            swarm_densitya.append(swarma_data[t]["density"])
            arg_lata.append(swarma_data[t]["argOfLat"])
        else:
            swarm_densitya.append(np.nan)
            arg_lata.append(np.nan)

    swarm_densitya = np.array(swarm_densitya)
    arg_lata = np.array(arg_lata)

    # Start Comparison
    swarmb_data = read_swarm(swarmb_file)
    swarm_densityb= []
    arg_latb = []
    for t in times:
        if t in swarmb_data:
            swarm_densityb.append(swarmb_data[t]["density"])
            arg_latb.append(swarmb_data[t]["argOfLat"])
        else:
            swarm_densityb.append(np.nan)
            arg_latb.append(np.nan)

    swarm_densityb = np.array(swarm_densityb)
    arg_latb = np.array(arg_latb)


    start_date = START_DATE
    end_date = END_DATE

   # Run MSIS
    moa_densitya, _ = run_msis(moa_file, swarma_data, model, start_date, end_date)
    moa_l_densitya, _ = run_msis(moa_l_file, swarma_data, model, start_date, end_date)
    moa_h_densitya, _ = run_msis(moa_h_file, swarma_data, model, start_date, end_date)
    msis_densitya, _ = run_msis(msis_file, swarma_data, model, start_date, end_date)

       # Run MSIS
    moa_densityb, _ = run_msis(moa_file, swarmb_data, model, start_date, end_date)
    moa_l_densityb, _ = run_msis(moa_l_file, swarmb_data, model, start_date, end_date)
    moa_h_densityb, _ = run_msis(moa_h_file, swarmb_data, model, start_date, end_date)
    msis_densityb, _ = run_msis(msis_file, swarmb_data, model, start_date, end_date)

    avg_timesa, swarm_avga, _ = compute_orbit_averages(arg_lata, swarm_densitya, times)
    avg_timesb, swarm_avgb, _ = compute_orbit_averages(arg_latb, swarm_densityb, times)


    _, moa_avga, _ = compute_orbit_averages(arg_lata, moa_densitya, times)
    _, moa_l_avga, _ = compute_orbit_averages(arg_lata, moa_l_densitya, times)
    _, moa_h_avga, _ = compute_orbit_averages(arg_lata, moa_h_densitya, times)
    _, msis_avga, _ = compute_orbit_averages(arg_lata, msis_densitya, times)

    _, moa_avgb, _ = compute_orbit_averages(arg_latb, moa_densityb, times)
    _, moa_l_avgb, _ = compute_orbit_averages(arg_latb, moa_l_densityb, times)
    _, moa_h_avgb, _ = compute_orbit_averages(arg_latb, moa_h_densityb, times)
    _, msis_avgb, _ = compute_orbit_averages(arg_latb, msis_densityb, times)


    plot_all(storm, clean_vals, avg_timesa=avg_timesa, avg_timesb=avg_timesb, 
             moa_avga=moa_avga, moa_avgb=moa_avgb, moa_uppera=moa_h_avga, moa_upperb=moa_h_avgb, 
             moa_lowera=moa_l_avga, moa_lowerb=moa_l_avgb, msis_avga=msis_avga, msis_avgb=msis_avgb,
             swarm_avga=swarm_avga, swarm_avgb=swarm_avgb, start_date=START_DATE, end_date=END_DATE)
