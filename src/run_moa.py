import subprocess
import sys
import os
import numpy as np
import argparse
from pathlib import Path
import json

if __name__ == '__main__':

    rel_path = "./" # Path to ouptut Folder
    src_path = "./" # Path to src Folder

    parser = argparse.ArgumentParser()
    parser.add_argument("event")
    parser.add_argument("start_date")
    parser.add_argument("end_date")
    parser.add_argument("--split", action="store_true")
    args = parser.parse_args()

    name = args.event
    start_date = np.datetime64(args.start_date)
    end_date = np.datetime64(args.end_date)

    assert (end_date - start_date) == np.timedelta64(10, "D"), (
        f"Expected a 10-day interval, got {end_date - start_date}"
    )

    mopt_date = start_date - np.timedelta64(20, 'D')
    mopt_mid_date = start_date - np.timedelta64(10, 'D')
    
    start_date_str_mopt = mopt_date.astype(str)
    end_date_str_mopt = start_date.astype(str)
    end_date_str_fopt = end_date.astype(str)
    end_date_cyg = (end_date + np.timedelta64(5, 'D')).astype(str)
    mopt_mid_str = mopt_mid_date.astype(str)

    print("===================================================")
    print(f"RUNNING MOA FOR {name} ({start_date} - {end_date})")
    print("===================================================")

    Path(f"{rel_path}/{name}").mkdir(parents=True, exist_ok=True)
    Path(f"{rel_path}/{name}/Results").mkdir(exist_ok=True)
    base_path = f"{rel_path}/{name}"

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

    subprocess.run(login_cmd, check=True)
    print("Login done, cookies saved.")

    if not args.split:
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

    else:
            
        url = (
            "https://www.space-track.org/basicspacedata/query/"
            "class/gp_history/"
            "SEMIMAJOR_AXIS/6728--6828/"
            f"EPOCH/{start_date_str_mopt}--{mopt_mid_str}/"
            "OBJECT_TYPE/PAYLOAD/"
            "orderby/TLE_LINE1%20ASC/format/json"
        )        
    
        if not os.path.exists(f"{base_path}/TLE_DATA_MOPT1.json"):

            with open(f"{base_path}/TLE_DATA_MOPT1.json", "w") as f:
                subprocess.run(
                    [
                        "curl",
                        "-b", f"{base_path}/cookies.txt",
                        url
                    ],
                    stdout=f,
                    check=True
                )

        print("Saved TLE_DATA_MOPT1.json")       

        url = (
            "https://www.space-track.org/basicspacedata/query/"
            "class/gp_history/"
            "SEMIMAJOR_AXIS/6728--6828/"
            f"EPOCH/{mopt_mid_str}--{end_date_str_mopt}/"
            "OBJECT_TYPE/PAYLOAD/"
            "orderby/TLE_LINE1%20ASC/format/json"
        )        
    
        if not os.path.exists(f"{base_path}/TLE_DATA_MOPT2.json"):

            with open(f"{base_path}/TLE_DATA_MOPT2.json", "w") as f:
                subprocess.run(
                    [
                        "curl",
                        "-b", f"{base_path}/cookies.txt",
                        url
                    ],
                    stdout=f,
                    check=True
                )

        print("Saved TLE_DATA_MOPT2.json")  

        with open(f"{base_path}/TLE_DATA_MOPT1.json", "r") as f:
            data1 = json.load(f)

        with open(f"{base_path}/TLE_DATA_MOPT2.json", "r") as f:
            data2 = json.load(f)

        combined = data1 + data2

        with open(f"{base_path}/TLE_DATA_MOPT.json", "w") as f:
            json.dump(combined, f)

        print("Saved TLE_DATA_MOPT.json")

    url = (
        "https://www.space-track.org/basicspacedata/query/"
        "class/gp_history/"
        "NORAD_CAT_ID/41884,41885,41886,41887,41888,41889,41890,41891/"
        f"EPOCH/{start_date_str_mopt}--{end_date_cyg}/"
        "orderby/TLE_LINE1%20ASC/format/json"
    )

    print(url)

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

    url = (
        "https://www.space-track.org/basicspacedata/query/"
        "class/gp_history/"
        "SEMIMAJOR_AXIS/6728--6828/"
        f"EPOCH/{end_date_str_mopt}--{end_date_str_fopt}/"
        "OBJECT_TYPE/PAYLOAD/"
        "orderby/TLE_LINE1%20ASC/format/json"
    )

    if not os.path.exists(f"{base_path}/TLE_DATA_FOPT.json"):

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

    if not os.path.exists(f"{base_path}/MOPT_OUTPUT.txt"):
        print("Running MOPT")
        subprocess.run(["python", f"{src_path}/mopt.py", name, "NRLMSISE00"])
        
        mass = {}
        with open(f"{base_path}/MOPT_OUTPUT.txt", "r") as f:
            for line in f:
                id, m = line.strip().split(",")
                id = id.strip()
                m = m.strip()
                mass[id] = float(m)
        
        ids = list(mass.keys())
        ids_str = ",".join(map(str, ids))

        url = (
            "https://www.space-track.org/basicspacedata/query/"
            "class/gp_history/"
            f"NORAD_CAT_ID/{ids_str}/"
            f"EPOCH/{end_date_str_mopt}--{end_date_str_fopt}/"
            "OBJECT_TYPE/PAYLOAD/"
            "orderby/TLE_LINE1%20ASC/format/json"
        )

        with open(f"{base_path}/TLE_DATA_FOPT.json", "w") as f:
            subprocess.run(["curl", "-b", f"{base_path}/cookies.txt", url],stdout=f, check=True)

        print("Saved TLE_DATA_FOPT.json")

        print("Running FOPT")
        subprocess.run(["python", f"{src_path}/fopt.py", name, "NRLMSISE00"])
