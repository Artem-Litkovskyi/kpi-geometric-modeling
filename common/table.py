import csv
import os


def reset_table(path, filename, column_names):
    """Create or reset a csv table."""
    os.makedirs(path, exist_ok=True)
    full_path = os.path.join(path, f"{filename}.csv")

    with open(full_path, "w", newline="", encoding="utf-8") as table:
        csv.writer(table).writerow(column_names)


def add_row_to_table(path, filename, column_values):
    """Add row into a csv table."""
    full_path = os.path.join(path, f"{filename}.csv")
    if not os.path.exists(full_path):
        raise ValueError(f"No table at this path: {full_path}")

    with open(full_path, "a", newline="", encoding="utf-8") as table:
        csv.writer(table).writerow(column_values)
