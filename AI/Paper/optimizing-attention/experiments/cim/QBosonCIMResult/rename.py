import os
import re

folder_path = './'

pattern_range = re.compile(r'OptimizingAttention(\d+)-(\d+)_(\d+).log')

for filename in os.listdir(folder_path):
    filepath = os.path.join(folder_path, filename)

    if not os.path.isfile(filepath) or not filepath.endswith('.log'):
        continue

    match_range = pattern_range.match(filename)
    start = int(match_range.group(1))
    index = int(match_range.group(3))
    new_number = start + (index - 1)
    new_name = f"OptimizingAttention{new_number}{os.path.splitext(filename)[1]}"
    os.rename(filepath, os.path.join(folder_path, new_name))
    print(f"Renamed: {filename} → {new_name}")
