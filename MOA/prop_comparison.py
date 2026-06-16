import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates 
from datetime import datetime
import subprocess
from collections import defaultdict
from matplotlib.ticker import ScalarFormatter
from load_gmat import *
import json
import pymsis 
import sys

def extract_elements_from_file(json_path, start_date, end_date):
    """
    Extracts all orbital elements for all spacecraft in a JSON file
    between start_date and end_date.
    """

    # Load JSON list
    with open(json_path, "r") as f:
        entries = json.load(f)

    elems_by_sc = defaultdict(list)

    for item in entries:
        epoch = np.datetime64(item["EPOCH"])

        if start_date <= epoch <= end_date:
            sc = item["OBJECT_ID"]
            elems_by_sc[sc].append(item)  # store full OMM entry!

    # Sort entries by epoch
    for sc, item_list in elems_by_sc.items():
        item_list.sort(key=lambda d: np.datetime64(d["EPOCH"]))

    return elems_by_sc

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python prop_comparison.py <month_year>")
        sys.exit(1)

    STORM = sys.argv[1]

    month_dict = {"MAR": "03", "APR": "04", "MAY": "05", "AUG": "08", "SEP": "09", "OCT": "10"}
    m,y = STORM.split('_')
    STORM_FIGURE_STR = f"{month_dict.get(m, None)}-{y}"

    DATE_FILE = f"/home/hennyc/data/{STORM}/DATES.txt"
    with open(DATE_FILE, "r") as file:
        _, _, _ = file.readline().strip().split(",")
        _, start_str, end_str = file.readline().strip().split(",")

    START_DATE = np.datetime64(start_str)
    END_DATE = np.datetime64(end_str)
    GMAT_FILE = "./msis_starter.script"
    msis_weather = "SpaceWeather-All-v1.2.txt"
    moa_weather = f"/home/hennyc/data/{STORM}/WEATHER.txt"

    elems = extract_elements_from_file("/home/hennyc/data/CYGNSS.json", START_DATE, END_DATE)
    filtered_sorted_tle_data = elems[f"2016-078D"]

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
        'WEATHERFILE_VAL': msis_weather,
        'LENGTH_VAL': round(((np.datetime64(filtered_sorted_tle_data[-1]['EPOCH']) - np.datetime64
                                (filtered_sorted_tle_data[0]['EPOCH'])) / np.timedelta64(1, 'D')),8)
    }

    with open(GMAT_FILE, 'r') as f:
        script = f.read()
    
    for key, val in values.items():
        script = script.replace(key, str(val))
            
    with open("msis.script", 'w') as f:
        f.write(script)

    try:
        gmat.LoadScript("msis.script")
        status = gmat.Execute()
        if status != 1:
            print("GMAT RUN FAILED")
            sys.exit()
    except Exception as e:
        print(e)
        sys.exit()
    finally:
        gmat.Clear()

    msis_output_data = np.loadtxt("/home/hennyc/src/msis_gmat_output.txt", delimiter=',')
    msis_time = msis_output_data[:, 0]
    msis_sma = msis_output_data[:,1]
    # msise90_densities = m90_output_data[:,2] * (10**5) #10^-5 kg/m^3
    # msise90_aop = m90_output_data[:,3]
    # msise90_ta = m90_output_data[:,4]
    # msise90_arglat = (msise90_aop + msise90_ta) % 360

    values.update({'WEATHERFILE_VAL': moa_weather})

    with open(GMAT_FILE, 'r') as f:
        script = f.read()
    
    for key, val in values.items():
        script = script.replace(key, str(val))
            
    with open("msis.script", 'w') as f:
        f.write(script)

    try:
        gmat.LoadScript("msis.script")
        status = gmat.Execute()
        if status != 1:
            print("GMAT RUN FAILED")
            sys.exit()
    except Exception as e:
        print(e)
        sys.exit()
    finally:
        gmat.Clear()

    moa_output_data = np.loadtxt("/home/hennyc/src/msis_gmat_output.txt", delimiter=',')
    moa_time = moa_output_data[:, 0]
    moa_sma = moa_output_data[:,1]
    # moa2_densities = moa2_output_data[:,2] * (10**5) #10^-5 kg/m^3
    # moa2_aop = moa2_output_data[:,3]
    # moa2_ta  = moa2_output_data[:,4]
    # moa2_arglat = (moa2_aop + moa2_ta) % 360

    # times = np.arange(START_DATE, END_DATE, np.timedelta64(30, "s"))
    # # swarm = Read_SWARM_Data(f"/home/hennyc/STORMS/{STORM}/DENSITY_DATA/SWARM{SWARM}.txt")
    # swarm_density = []
    # arg_lat = []
    # for t in times:
    #     if t in swarm:
    #         swarm_density.append(swarm[t]["density"] * 1e9)
    #         arg_lat.append(swarm[t]["argOfLat"])
    #     else:
    #         swarm_density.append(np.nan)
    #         arg_lat.append(np.nan)

    # swarm_density = np.array(swarm_density) * (10**5)
    # arg_lat = np.array(arg_lat)

    tle_epoch = np.datetime64(datetime.strptime(filtered_sorted_tle_data[0]['EPOCH'], "%Y-%m-%dT%H:%M:%S.%f"))
    msis_times_dt = tle_epoch + msis_time.astype('timedelta64[s]')
    moa_times_dt   = tle_epoch + moa_time.astype('timedelta64[s]')

    # swarm_orbit_times, swarm_orbit_dens = compute_orbit_averages(
    #     times, 
    #     arg_lat, 
    #     swarm_density
    # )

    # msis90_orbit_times, msis90_orbit_dens = compute_orbit_averages(
    #     msis90_times_dt,
    #     msise90_arglat,
    #     msise90_densities
    # )

    # moa2_orbit_times, moa2_orbit_dens = compute_orbit_averages(
    #     moa2_times_dt,
    #     moa2_arglat,
    #     moa2_densities
    # )

    # plt.figure(figsize=(8,6))

    # plt.plot(swarm_orbit_times, swarm_orbit_dens, 'k-', label="Swarm Accelerometer")
    # plt.plot(msis90_orbit_times, msis90_orbit_dens, 'r-', label="MSISE90")
    # plt.plot(moa2_orbit_times,  moa2_orbit_dens,  'b-', label="MOA 2")

    # plt.title(f"Orbit-Averaged Density — SWARM {SWARM}")
    # plt.xlabel("Time")
    # plt.ylabel("Density")
    # plt.grid(True)
    # plt.legend()
    # plt.gcf().autofmt_xdate()
    # plt.savefig(f"/home/hennyc/STORMS/{STORM}/Results/Swarm{SWARM}_densities.png")
    # plt.close()

    sma_times_swarm = np.array([np.datetime64(e['EPOCH']) for e in filtered_sorted_tle_data])
    sma_swarm = np.array([float(e['SEMIMAJOR_AXIS']) for e in filtered_sorted_tle_data])

    # msis90_orbit_t, msis90_orbit_sma = compute_orbit_averages(
    #     msis90_times_dt,
    #     msise90_arglat,
    #     msis90_sma
    # )

    # moa2_orbit_t, moa2_orbit_sma = compute_orbit_averages(
    #     moa2_times_dt,
    #     moa2_arglat,
    #     moa2_sma
    # )

    fig = plt.figure(figsize=(12,6))
    ax = plt.gca()
    size = 14

    # # Get the center x-position of the axes in figure coordinates
    # bbox = ax.get_position()
    # x_center = bbox.x0 + bbox.width / 2

    # # Put title at "suptitle height" (e.g., y=0.95)
    # fig.text(x_center, 0.98, f"{STORM_FIGURE_STR}: Propogation of CYGNSS A Spacecraft",
    #         ha='center', va='top', fontsize=17, fontweight='bold')

    ax.plot(msis_times_dt, msis_sma, 'b-', linewidth=2, label="NRLMSISE-00")
    ax.plot(moa_times_dt,  moa_sma,  'r-', linewidth=2, label="MOA")
    ax.plot(sma_times_swarm, sma_swarm, 'k-', marker='D',linewidth=5, markersize=14, label=f"TLEs")

    # np.save(f"msis_data/{STORM}_tle_times.npy", sma_times_swarm)
    # np.save(f"msis_data/{STORM}_tle_sma.npy", sma_swarm)
    # np.save(f"msis_data/{STORM}_moa_times.npy", moa2_times_dt)
    # np.save(f"msis_data/{STORM}_moa_sma.npy", moa2_sma)
    # np.save(f"msis_data/{STORM}_msise_times.npy", msis90_times_dt)
    # np.save(f"msis_data/{STORM}_msise_sma.npy", msis90_sma)

    values = {"msis": msis_sma[-1], "moa": moa_sma[-1], "swarm": sma_swarm[-1]}
    ranked = sorted(values, key=values.get, reverse=True)

    ypos = {}
    ypos[ranked[0]] = 20
    ypos[ranked[1]] = -20
    ypos[ranked[2]] = 0

    y_m9 = ypos["msis"]
    y_m = ypos["moa"]
    y_s = ypos["swarm"]
    x1 = 155

    # --- MSISE90 ---
    x_msis = msis_times_dt[-1]
    y_msis = msis_sma[-1]
    plt.annotate(
        f"{y_msis:.2f} km",
        (x_msis, y_msis),
        textcoords="offset points",
        xytext=(x1, y_m9),        # offset position
        ha='right',
        fontsize=size,
        fontweight='bold',
        color='blue',
        bbox=dict(boxstyle="square", fc="white", ec="blue", alpha=1),
        arrowprops=dict(arrowstyle="->", color='blue', lw=2)
    )

    # --- MOA 2 ---
    x_moa = moa_times_dt[-1]
    y_moa = moa_sma[-1]
    plt.annotate(
        f"{y_moa:.2f} km",
        (x_moa, y_moa),
        textcoords="offset points",
        xytext=(x1, y_m),        # offset down-right
        ha='right',
        fontsize=size,
        fontweight='bold',
        color='red',
        bbox=dict(boxstyle="square", fc="white", ec="red", alpha=1),
        arrowprops=dict(arrowstyle="->", color='red', lw=2)
    )

    # --- SWARM ---
    x_swarm = sma_times_swarm[-1]
    y_swarm = sma_swarm[-1]
    plt.annotate(
        f"{y_swarm:.2f} km",
        (x_swarm, y_swarm),
        textcoords="offset points",
        xytext=(x1, y_s),        # offset left and up
        ha='right',
        fontsize=size,
        fontweight='bold',
        color='black',
        bbox=dict(boxstyle="square", fc="white", ec="black", alpha=1),
        arrowprops=dict(arrowstyle="->", color='black', lw=2)
    )



    #ax.set_title(f"{STORM_FIGURE_STR}: Propogation of CYGNSS A Spacecraft", fontweight='bold', fontsize=17)
    ax.set_xlabel("Real Time", fontweight='bold', fontsize=size)
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.yaxis.get_major_formatter().set_useOffset(False)
    ax.set_ylabel("SMA(km)", fontweight='bold', fontsize=size)
    leg = ax.legend(loc="upper left", bbox_to_anchor=(1.015, 1), frameon=False, fontsize=size, handlelength = 1.5) # bbox_to_anchor=(1.02, 1)
    for line in leg.get_lines():
        line.set_linewidth(3)
    ax.tick_params(labelsize=size)
    ax.set_xlim(START_DATE, END_DATE)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    plt.tight_layout()
    fig.savefig(f"/home/hennyc/src/oct_prop.png")
    plt.close()

    # print("min/max msise90_arglat:", np.nanmin(msise90_arglat), np.nanmax(msise90_arglat))
    # print("min/max moa2_arglat:", np.nanmin(moa2_arglat), np.nanmax(moa2_arglat))
    # print("msise90_orbits:", len(msis90_orbit_times))
    # print("moa2_orbits:", len(moa2_orbit_times))
    # print("swarm_orbits:", len(swarm_orbit_times))
