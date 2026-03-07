from sleepjepa.slumber import edf_signals_to_zarr, HSP_CHANNEL_NAME_MAP

import multiprocessing as mp
from pathlib import Path
import glob
import time
import dask.array as da

import pandas as pd

write_data_dir = Path("human_sleep_project_waveforms_no_resampling/")
edf_files = glob.glob('HSP_sleep_data/bids/**/**/eeg/*.edf')
current_zarr_files = glob.glob(str(write_data_dir/"*.zarr"))

HSP_CHANNEL_NAME_MAP = {'ABD':'ABDO RES',
                        'ABDOMEN':'ABDO RES',
                        'Abdomen':'ABDO RES',
                        'CHEST':'THOR RES',
                        'Chest':'THOR RES',
                        'THORAX':'THOR RES',
                        'C4-M1':'C4-M1',
                        'C4-AVG':'C4-AVG',
                        'C3-M2':'C3-M2',
                        'C3-AVG':'C3-AVG',
                        'E1-M2':'EOG(L)',
                        'E1-AVG':'E1-AVG',
                        'C4':'C4',
                        'M1':'M1',
                        'C3':'C3',
                        'M2':'M2',
                        'E1':'E1',
                        'EKG':'ECG',
                        'ECG':'ECG',
                        'ECG-LL':'ECG (LL)',
                        'ECG-RA':'ECG (RA)',
                        'ECG-V1':'ECG (V1)',
                        'SaO2':'SaO2',
                        'SpO2':'SaO2',
                        'SPO2':'SaO2',
                        'EMG':'EMG',
                        'CHIN':'EMG (chin)',
                        'CHIN1-CHIN2':'EMG (1-2)',
                        'Chin1-Chin2':'EMG (1-2)',
                        'Chin1-Chin3':'EMG (1-3)',
                        'CHIN1-CHIN3':'EMG (1-3)',
                        'CHIN3':'EMG (3)'
                        }

channels = list(HSP_CHANNEL_NAME_MAP.keys())

df = pd.DataFrame(edf_files, columns=['file_path'])
df['file_name'] = df['file_path'].apply(lambda x: Path(x).stem)
df['zarr_exists'] = df['file_name'].isin(map(lambda x: Path(x).stem, current_zarr_files))
edf_files = df.loc[df['zarr_exists'] == False, 'file_path'].unique().tolist()

def subtract_channels(rt_grp, left_channel, right_channel, new_name):
    signal_right = rt_grp[right_channel][:]
    signal_left = rt_grp[left_channel][:]
    signal_sub = signal_left - signal_right
    a = da.from_array(signal_sub, chunks='auto')
    a.to_zarr(url=rt_grp.store, component=new_name, compute=True)
    name = mapped_label = new_name
    header = {}
    header['label'] = name
    header['mapped_label'] = mapped_label
    for i in ['digital_max', 'digital_min', 'dimension', 'prefilter', 'sample_frequency', 'sample_rate']:
        assert rt_grp[left_channel].attrs['signal_header'][i] == rt_grp[right_channel].attrs['signal_header'][i]
        header[i] = rt_grp[left_channel].attrs['signal_header'][i]
    header['physical_max'] = rt_grp[left_channel].attrs['signal_header']['physical_max'] - rt_grp[right_channel].attrs['signal_header']['physical_max']
    header['physical_min'] = rt_grp[left_channel].attrs['signal_header']['physical_min'] - rt_grp[right_channel].attrs['signal_header']['physical_min']
    rt_grp[name].attrs['signal_header'] = header

def main_function(file, frequency=None, channels=channels, channel_name_map=HSP_CHANNEL_NAME_MAP, write_data_dir=write_data_dir):
    try:
        rt_grp = edf_signals_to_zarr(file, frequency=frequency, channels=channels, channel_name_map=channel_name_map, write_data_dir=write_data_dir)
        if 'ECG (RA)' in rt_grp.array_keys() and 'ECG (LL)' in rt_grp.array_keys():
            right_name = 'ECG (RA)'
            left_name = 'ECG (LL)'
            subtract_channels(rt_grp, left_channel=left_name, right_channel=right_name, new_name='ECG (LL-RA)')
        if ('C4' in rt_grp.array_keys()) and ('M1' in rt_grp.array_keys()) and ('C4-M1' not in rt_grp.array_keys()):
            right_name = 'M1'
            left_name = 'C4'
            subtract_channels(rt_grp, left_channel=left_name, right_channel=right_name, new_name='C4-M1')
        if ('C3' in rt_grp.array_keys()) and ('M2' in rt_grp.array_keys()) and ('C3-M2' not in rt_grp.array_keys()):
            right_name = 'M2'
            left_name = 'C3'
            subtract_channels(rt_grp, left_channel=left_name, right_channel=right_name, new_name='C3-M2')
        if ('E1' in rt_grp.array_keys()) and ('M2' in rt_grp.array_keys()) and ('E1-M2' not in rt_grp.array_keys()):
            right_name = 'M2'
            left_name = 'E1'
            subtract_channels(rt_grp, left_channel=left_name, right_channel=right_name, new_name='EOG(L)')
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