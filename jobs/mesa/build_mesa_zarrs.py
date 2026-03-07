from sleepjepa.slumber import edf_signals_to_zarr

import multiprocessing as mp
from pathlib import Path
import glob
import time

import pandas as pd

write_data_dir = Path("mesa_waveforms_no_resampling/")
edf_files = glob.glob('MESA/*/*.edf')
current_zarr_files = glob.glob(str(write_data_dir/"*.zarr"))

df = pd.DataFrame(edf_files, columns=['file_path'])
df['file_name'] = df['file_path'].apply(lambda x: Path(x).stem)
df['zarr_exists'] = df['file_name'].isin(map(lambda x: Path(x).stem, current_zarr_files))
edf_files = df.loc[df['zarr_exists'] == False, 'file_path'].unique().tolist()

def main_function(file, frequency=None, write_data_dir=write_data_dir):
    try:
        _ = edf_signals_to_zarr(file, frequency=frequency, write_data_dir=write_data_dir)
    except Exception as e:
        print(f"Error parsing file: {file}. Error: {e}.", flush=True)


if __name__ == '__main__':
    print(f'Beginning MP Job with {mp.cpu_count()} processes')
    start_time = time.time()
    with mp.Pool(12) as pool:
        result = pool.map_async(main_function, edf_files)
        pool.close()
        pool.join()
    print('Job Completed')
    print(f"--- {time.time() - start_time} seconds ---")
