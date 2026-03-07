import glob
import pandas as pd
import time
import multiprocessing as mp
from pathlib import Path

hypnogram_tables = glob.glob('MrOS_data/polysomnography/annotations-events-nsrr/*/*.xml')
write_data_dir = Path("mros_waveforms_no_resampling/hypnogram_csvs/")

map_ = {'Wake|0':0, 'Stage 1 sleep|1':1, 'Stage 2 sleep|2':2, 'Stage 3 sleep|3':3, 'Stage 4 sleep|4':3, 'REM sleep|5':4, 'Unscored|9':-100}

def main_function(idx,
                  file,
                  ):
    try:
        filename = Path(file)
        df = pd.read_xml(file, xpath='.//ScoredEvent')
        df = df.loc[df['EventConcept'].isin(list(map_.keys()))]
        df['shift_duration'] = df['Start'].diff().shift(-1)
        df['num_repeats'] = df['shift_duration'] / 30
        df.loc[df.num_repeats.isna(), 'num_repeats'] = df.loc[df.num_repeats.isna(), 'Duration'] / 30
        df = df.reindex(df.index.repeat(df.num_repeats))
        df[1] = df['EventConcept'].map(map_)
        file_stem = filename.stem.strip('-nsrr') + '-hyp.csv'
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