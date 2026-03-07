from sleepjepa.slumber import edf_signals_to_zarr

import multiprocessing as mp
from pathlib import Path
import glob
import time
import dask.array as da
import pandas as pd

write_data_dir = Path("mnc_waveforms_no_resampling/")
hyp_data_dir = Path("mnc_waveforms_no_resampling/hypnogram_csvs/")
edf_files = glob.glob('mnc/**/*.edf')
current_zarr_files = glob.glob(str(write_data_dir/"*.zarr"))

MNC_CHANNEL_NAME_MAP = {
         'cs_LOC':'LOC',
         'cs_ROC':'ROC',
         'cs_ECG':'ECG',
         'cs_EEG':'C4-M1',
         'cs_EMG':'EMG',
         'cchin':'cchin',
         'chin':'chin',
         'cchin_l':'cchin_l',
         'thorax':'thorax',
         'abdomen':'abdomen',
         'spo2':'spo2',
         'flow':'flow',
         'pleth':'pleth',
         }

channels = list(MNC_CHANNEL_NAME_MAP.keys())

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
    for i in ['digital_max', 'digital_min', 'dimension', 'prefilter', 'sample_frequency']:
        assert rt_grp[left_channel].attrs['signal_header'][i] == rt_grp[right_channel].attrs['signal_header'][i]
        header[i] = rt_grp[left_channel].attrs['signal_header'][i]
    header['physical_max'] = rt_grp[left_channel].attrs['signal_header']['physical_max'] - rt_grp[right_channel].attrs['signal_header']['physical_max']
    header['physical_min'] = rt_grp[left_channel].attrs['signal_header']['physical_min'] - rt_grp[right_channel].attrs['signal_header']['physical_min']
    rt_grp[name].attrs['signal_header'] = header

def main_function(file, frequency=None, channels=channels, channel_name_map=MNC_CHANNEL_NAME_MAP, write_data_dir=write_data_dir):
    try:
        rt_grp = edf_signals_to_zarr(file, frequency=frequency, channels=channels, channel_name_map=channel_name_map, write_data_dir=write_data_dir, hyp_epoch_length=1, hyp_data_dir=hyp_data_dir)
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
