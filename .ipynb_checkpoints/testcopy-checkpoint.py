import os
import shutil
import random
import re

def copy_files_based_on_galah_id(source_dir, destination_dir, num_samples=300):
    # Use a reservoir sampling algorithm to randomly select files from a large directory
    selected_files = []
    for i, file_name in enumerate(os.listdir(source_dir)):
        if os.path.isfile(os.path.join(source_dir, file_name)):
            if i < num_samples:
                selected_files.append(file_name)
            else:
                r = random.randint(0, i)
                if r < num_samples:
                    selected_files[r] = file_name

    # Now find and copy files with the matching galah_id
    for file in selected_files:
        # Extract the first 15 digits from the filename
        match = re.match(r"(\d{15})", file)
        if match:
            galah_id = match.group(1)
            # Copy files starting with this galah_id
            for f in os.listdir(source_dir):
                if f.startswith(galah_id):
                    src = os.path.join(source_dir, f)
                    dest = os.path.join(destination_dir, f)
                    shutil.copy2(src, dest)
                    print(f"Copied {f} to {destination_dir}")



if __name__ == "__main__":
    source_dir = '../../../projects/k-pop/spectra/galah/dr3'
    # Destination directory
    dest_dir = 'data/test2/galah'
    copy_files_based_on_galah_id(source_dir, dest_dir)
