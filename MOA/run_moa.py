import subprocess
import sys
import os
import numpy as np
import argparse
from pathlib import Path

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("event")
    parser.add_argument("start_date")
    parser.add_argument("end_date")
    parser.add_argument("--afrl", action="store_true")
    args = parser.parse_args()

    name = args.event
    start_date = np.datetime64(args.start_date)
    end_date = np.datetime64(args.end_date)
    afrl = args.afrl

    mopt_date = start_date - np.timedelta64(20, 'D')
    
    start_date_str_mopt = mopt_date.astype(str)
    end_date_str_mopt = start_date.astype(str)
    end_date_str_fopt = end_date.astype(str)

    print(f"RUNNING MOA FOR {name}")

    if afrl:
        Path(f"/home/hennyc/afrl/moa/{name}").mkdir(parents=True, exist_ok=True)
        Path(f"/home/hennyc/afrl/moa/{name}/Results").mkdir(exist_ok=True)
        base_path = f"/home/hennyc/afrl/moa/{name}"
    else:
        Path(f"/home/hennyc/data/{name}").mkdir(parents=True, exist_ok=True)
        Path(f"/home/hennyc/data/{name}/Results").mkdir(exist_ok=True)
        base_path = f"/home/hennyc/data/{name}"

    if not os.path.exists(f"{base_path}/DATES.txt"):
        with open(f"{base_path}/DATES.txt", "w") as f:
            f.write(f"MOPT,{mopt_date},{start_date}\n")
            f.write(f"FOPT,{start_date},{end_date}")
    
    login_cmd = [
        "curl",
        "-c", f"{base_path}/cookies.txt",
        "-b", f"{base_path}/cookies.txt",
        "https://www.space-track.org/ajaxauth/login",
        "-d", "identity=hennyc@umich.edu&password=Michigan92!Ff121f89"
    ]

    if not os.path.exists(f"{base_path}/cookies.txt"):
        subprocess.run(login_cmd, check=True)
    print("Login done, cookies saved.")

    url = (
        "https://www.space-track.org/basicspacedata/query/"
        "class/gp_history/"
        "SEMIMAJOR_AXIS/6728--6828/"
        f"EPOCH/{start_date_str_mopt}--{end_date_str_mopt}/"
        "OBJECT_TYPE/PAYLOAD/"
        "orderby/TLE_LINE1%20ASC/format/json"
    )

    if not os.path.exists(f"{base_path}/TLE_DATA_MOPT.json"):

        with open(f"{base_path}/TLE_DATA_MOPT.json", "w") as f:
            subprocess.run(
                [
                    "curl",
                    "-b", f"{base_path}/cookies.txt",
                    url
                ],
                stdout=f,
                check=True
            )

    print("Saved TLE_DATA_MOPT.json")



    url = (
        "https://www.space-track.org/basicspacedata/query/"
        "class/gp_history/"
        "SEMIMAJOR_AXIS/6728--6828/"
        f"EPOCH/{end_date_str_mopt}--{end_date_str_fopt}/"
        "OBJECT_TYPE/PAYLOAD/"
        "orderby/TLE_LINE1%20ASC/format/json"
    )

    if not os.path.exists(f"{base_path}/TLE_DATA_MOPT.json"):

        with open(f"{base_path}/TLE_DATA_FOPT.json", "w") as f:
            subprocess.run(
                [
                    "curl",
                    "-b", f"{base_path}/cookies.txt",
                    url
                ],
                stdout=f,
                check=True
            )

    print("Saved TLE_DATA_FOPT.json")



    url = (
        "https://www.space-track.org/basicspacedata/query/"
        "class/gp_history/"
        "NORAD_CAT_ID/41884,41885,41886,41887,41888,41889,41890,41891/"
        f"EPOCH/{start_date_str_mopt}--{end_date_str_fopt}/"
        "orderby/TLE_LINE1%20ASC/format/json"
    )

    if not os.path.exists(f"{base_path}/CYGNSS.json"):

        with open(f"{base_path}/CYGNSS.json", "w") as f:
            subprocess.run(
                [
                    "curl",
                    "-b", f"{base_path}/cookies.txt",
                    url
                ],
                stdout=f,
                check=True
            )

    print("Saved CYGNSS.json")



    if afrl:
        print("Running MOPT")
        subprocess.run([
            "python",
            "/home/hennyc/src/mopt.py",
            name,
            "NRLMSISE00",
            "--afrl"
        ])

        print("Running FOPT")
        subprocess.run([
            "python",
            "/home/hennyc/src/fopt.py",
            name,
            "NRLMSISE00",
            "--afrl"
        ])
    else:
        print("Running MOPT")
        subprocess.run([
            "python",
            "/home/hennyc/src/mopt.py",
            name,
            "NRLMSISE00"
        ])

        print("Running FOPT")
        subprocess.run([
            "python",
            "/home/hennyc/src/fopt.py",
            name,
            "NRLMSISE00"
        ])
