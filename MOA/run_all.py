import subprocess
import threading

storms = {
    "APR_2023": ("2023-04-19", "2023-04-29"),
    "OCT_2024": ("2024-10-05", "2024-10-15"),
    "SEP_2024": ("2024-09-10", "2024-09-20"),
    "MAY_2024": ("2024-05-05", "2024-05-15"),
    "MAR_2024": ("2024-02-27", "2024-03-08"),
    "APR_2024": ("2024-04-14", "2024-04-24"),
    "AUG_2024": ("2024-08-08", "2024-08-18"),
    "MAR_2023": ("2023-03-19", "2023-03-29")
}   # "MAY_2017": ("2017-05-23", "2017-06-02")

def input_with_timeout(prompt, timeout=2, default="y"):
    user_input = [default]

    def ask():
        try:
            val = input(prompt).strip().lower()
            if val:
                user_input[0] = val
        except EOFError:
            pass

    thread = threading.Thread(target=ask)
    thread.daemon = True
    thread.start()
    thread.join(timeout)

    return user_input[0]

for storm, (start_date_str, end_date_str) in storms.items():
    print(f"\n=== Processing storm: {storm} ===")

    commands = [
        # ["python", "/home/hennyc/afrl/run_moa.py", storm, start_date_str, end_date_str],
        ["python", "mopt.py", storm, "NRLMSISE00"],
        ["python", "fopt.py", storm, "NRLMSISE00"],
        ["python", "run_msis.py", storm, "A", "NRLMSISE00"],
        ["python", "run_msis.py", storm, "B", "NRLMSISE00"],
        ["python", "visualizer.py", storm, "A", "NRLMSISE00"]
    ]

    for cmd in commands:
        print(f"Running {cmd}")
        result = subprocess.run(cmd)

        if result.returncode != 0:
            print("Command failed, stopping this storm.")
            break

    # ---- ask user whether to continue (with timeout) ----
    user_input = input_with_timeout(
        f"Finished {storm}. Continue to next storm? (y/n): ",
        timeout=10,
        default="y"
    )

    if user_input not in ["y", "yes"]:
        print("Stopping execution.")
        break

print("Done.")
