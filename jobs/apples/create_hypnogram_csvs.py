import glob
import pandas as pd
import time
import multiprocessing as mp
from pathlib import Path

hypnogram_tables = glob.glob('/apples/polysomnography/*/*.annot')
write_data_dir = Path("apples_waveforms_no_resampling/hypnogram_csvs/")

map_ = {'W':0, 'N1':1, 'N2':2, 'N3':3, 'R':4}

def main_function(idx,
                  file,
                  ):
    try:
        filename = Path(file)
        df = pd.read_table(file)
        df = df.loc[df['class'].isin(['W','N1','N2','N3','R'])]
        df[1] = df['class'].map(map_)
        file_stem = filename.stem + '-hyp.csv'
        df.to_csv(write_data_dir/Path(file_stem), index=False)
    except Exception as e:
        print(f"Error parsing file: {file}. Error: {e}.", flush=True)
    

if __name__ == '__main__':
    print(f'Beginning MP Job with {mp.cpu_count()} processes')
    start_time = time.time()
    with mp.Pool() as pool:
        result = pool.starmap_async(main_function, enumerate(hypnogram_tables))
        pool.close()
        pool.join()
    print('Job Completed')
    print(f"--- {time.time() - start_time} seconds ---")