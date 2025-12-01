import csv
import os

def update_csv(csv_path, file_name, time_taken):
    rows = {}

    # Load existing rows if CSV exists
    if os.path.exists(csv_path):
        with open(csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                file_num = int(row["file_num"])
                rows[file_num] = {
                    "file_name": row["file_name"],
                    "time_taken": float(row["time_taken"])
                }

    file_num = int(file_name) # same
    
    # Update current file's entry
    rows[file_num] = {
        "file_name": file_name,
        "time_taken": time_taken
    }

    # Write all rows back to CSV
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["file_num", "file_name", "time_taken"])
        writer.writeheader()
        for num in sorted(rows.keys()):
            writer.writerow({
                "file_num": num,
                "file_name": rows[num]["file_name"],
                "time_taken": rows[num]["time_taken"]
            })