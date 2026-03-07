import torch, pandas as pd, os, glob, re, lightning.pytorch as pl
import yaml
from sleepjepa.survival import PatchTFTSurvivalDemo,  make_cuts_days
from sleepjepa.slumber import SingleOutcomeDataset, ALL_FREQUENCY_FILTERS, ALL_CHANNELS, nested_tensor_sequence_multi_label_collate
from sleepjepa.jepa import JEPASimpleLightning, LeJEPALightning2
from sleepjepa.nested import get_predictions_nested_survival
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader
from datetime import datetime

from sleepjepa.heads import AttentiveClassifier
from sleepjepa.augmentations import VariableChannelInput
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from pathlib import Path
import torch.nn as nn
import sys

torch.set_float32_matmul_precision('high')
torch.backends.cuda.enable_flash_sdp(True)

outcome_idx = sys.argv[1] if len(sys.argv) > 1 else None
config_path = 'config/train_sleep_long_term_outcome.yaml'
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

outcome_config_path = 'outcome_path_map.yaml'
with open(outcome_config_path, 'r') as f:
    outcome_paths = yaml.safe_load(f)

outcomes = list(outcome_paths.keys())
if outcome_idx is not None:
    outcome_idx = int(outcome_idx) - 1
    assert outcome_idx >=0 and outcome_idx < len(outcomes), f"Outcome index {outcome_idx} out of range."
    outcome_cols = outcomes[outcome_idx]
    config['run_config']['outcome_cols'] = [outcome_cols]

outcome_cols = config['run_config']['outcome_cols']
time_cols = [f"{col}_time" for col in outcome_cols]

config['paths'].update(outcome_paths[outcome_cols[0]])

n_demographic_embeddings = config['dataset'].get('demographic_embeddings', [])
y_demographic_columns = config['dataset'].get('y_demographic_columns', [])
demographic_embeddings = dict(zip(y_demographic_columns, n_demographic_embeddings))

if not config['dataset'].get('demographic_predictor', False):
    demographic_embeddings = {}

dataset_filename = config['paths']['dataset_filename']
models_dir = config['paths']['models_dir']

pretrained_encoder_path = os.path.join(config['paths']['encoder_dir'], config['paths']['pretrained_encoder_path'])

try:
    encoder_model = JEPASimpleLightning.load_from_checkpoint(pretrained_encoder_path, map_location='cpu')
    num_heads = encoder_model.encoder.predictor_blocks[0].self_attn.num_heads
    d_model = encoder_model.encoder.d_model
except:
    encoder_model = LeJEPALightning2.load_from_checkpoint(pretrained_encoder_path, map_location='cpu')
    num_heads = encoder_model.encoder.predictor_blocks[0].self_attn.num_heads
    d_model = encoder_model.encoder.d_model


shhs_file_path = config['paths'].get('shhs_file_path', None)
mros_file_path = config['paths'].get('mros_file_path', None)
mesa_file_path = config['paths'].get('mesa_file_path', None)
wsc_file_path = config['paths'].get('wsc_file_path', None)
ds = ['SHHS', 'MROS', 'MESA', 'WSC']

outcome_dfs = []
ext_test_dfs = []
for n,f in zip(ds, [shhs_file_path, mros_file_path, mesa_file_path, wsc_file_path]):
    if f is not None:
        assert os.path.exists(f), f"File {f} does not exist."
        df = pd.read_csv(f)
        if n in config['paths']['ext_test_sets']:
            ext_test_dfs.append(df)
        else:
            outcome_dfs.append(df)
    else:
        continue
outcome_data = pd.concat(outcome_dfs).reset_index(drop=True)
if len(ext_test_dfs) > 0:
    ext_outcome_df = pd.concat(ext_test_dfs).reset_index(drop=True)

train_zarrs, val_zarrs, test_zarrs = [], [], []
splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=config['run_config']['random_state'])
labels = outcome_data[outcome_cols[0]].values.tolist()
groups = [Path(i).stem for i in outcome_data.filepath.unique()]

if config['paths']['split_shhs_sleep_fm']:
    shhs_test_ids = pd.read_csv(config['paths']['shhs_sleepfm_ids'])['nsrrid'].unique().tolist()
    test_zarrs.extend(outcome_data[outcome_data['nsrrid'].isin(shhs_test_ids)]['filepath'].unique().tolist())
    df = outcome_data[~outcome_data['nsrrid'].isin(shhs_test_ids)].reset_index(drop=True)
    train_val_zarrs = outcome_data.filepath.unique().tolist()
else:
    train_idxs, test_idxs = next(splitter.split(X=outcome_data.filepath.unique(),  y=labels, groups=groups))
    train_val_zarrs = [outcome_data.filepath.unique()[i] for i in train_idxs]
    test_zarrs.extend([outcome_data.filepath.unique()[i] for i in test_idxs])
labels = outcome_data[outcome_data['filepath'].isin(train_val_zarrs)][outcome_cols[0]].values.tolist()
groups = [Path(i).stem for i in train_val_zarrs]
splitter2 = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=config['run_config']['random_state'])
train_idxs, val_idxs = next(splitter2.split(X=train_val_zarrs,  y=labels, groups=groups))
train_zarrs.extend([train_val_zarrs[i] for i in train_idxs])
val_zarrs.extend([train_val_zarrs[i] for i in val_idxs])

print(f"Number of training samples: {len(train_zarrs)}")
print(f"Number of validation samples: {len(val_zarrs)}")
print(f"Number of test samples: {len(test_zarrs)}")
if len(ext_test_dfs) > 0:
    print(f"Number of external test samples: {len(ext_outcome_df.filepath.unique().tolist())}")

if not config['run_config']['do_train'] and len(config['run_config']['remove_channel_idxs']) > 0:
    vb_callback = VariableChannelInput(
        indexes_to_add_channels=config['run_config']['remove_channel_idxs'], 
        n_channels_expected=len(ALL_CHANNELS),
        channel_dim=1
    )
    channels = [ch for i,ch in enumerate(ALL_CHANNELS) if i not in config['run_config']['remove_channel_idxs']]
    c_in= len(ALL_CHANNELS)
    encoder_model.encoder.tokenizer.variable_channel_callback = vb_callback
else:
    channels = ALL_CHANNELS
    c_in = len(channels)

frequency = config['dataset']['frequency']

win_length=frequency*config['dataset']['patch_seconds'] 
hop_length=win_length - int(config['dataset']['overlap']*win_length)
max_seq_len_sec = config['dataset']['max_seq_len_sec']
min_seq_len_sec = config['dataset']['min_seq_len_sec']
seq_len_sec = sample_stride = max_seq_len_sec
max_seq_len = seq_len_sec*frequency if seq_len_sec is not None else max_seq_len_sec*frequency
n_patches = (max(max_seq_len, win_length)-win_length) // hop_length + 1
if ((max_seq_len-win_length) % hop_length != 0):
    n_patches += 1

num_classes = len(make_cuts_days(min_year=config['training']['time_range_years'][0], max_year=config['training']['time_range_years'][1], bin_years=config['training']['discrete_time_bins']))

lp_head = dict(embed_dim=d_model,
        num_heads=config['lp_head'].get('num_heads', num_heads),
        mlp_ratio=config['lp_head']['mlp_ratio'],
        depth=config['lp_head']['depth'],
        c_in=c_in,
        norm_layer=nn.LayerNorm,
        init_std=config['lp_head']['init_std'],
        qkv_bias=config['lp_head']['qkv_bias'],
        num_classes=num_classes,
        complete_block=config['lp_head']['complete_block'],
        per_channel=config['lp_head'].get('per_channel', False)
    )

lp_model = AttentiveClassifier(**lp_head)

scheduler_kwargs = config['scheduler'].copy()
del scheduler_kwargs['scheduler_type']

filename = f"{config['run_config']['name']}-{outcome_cols[0]}" + "{epoch:02d}-mean_ipcw_auc-{mean_ipcw_auc:.5f}-brier_score-{brier_score:.5f}-cindex_ipcw-{cindex_ipcw:.5f}-all_score-{all_score:.5f}"
date_str = datetime.now().strftime("%Y-%m-%d")
if config['run_config']['do_train']:
    sub_dir = f"{date_str}-{config['run_config']['model_type']}-{config['run_config']['model_run']}-{outcome_cols[0]}" 
else:
    sub_dir = outcome_paths[outcome_cols[0]]['reps_demographics_best_path'] if config['dataset'].get('demographic_predictor', False) else outcome_paths[outcome_cols[0]]['no_demos_best_path']
    all_checkpoints = glob.glob(os.path.join(models_dir, sub_dir, "*.ckpt"))
    pattern = re.compile(r"all_score=(\d+\.\d+)")
    all_checkpoints_scores = [(ckpt, float(pattern.search(ckpt).group(1))) for ckpt in all_checkpoints if pattern.search(ckpt)]
    all_checkpoints_scores.sort(key=lambda x: x[1], reverse=True)
    best_model_path = all_checkpoints_scores[0][0]

checkpoint_callback1 = ModelCheckpoint(dirpath=os.path.join(models_dir, sub_dir), save_top_k=1, monitor="mean_ipcw_auc", mode='max', filename=filename)
checkpoint_callback2 = ModelCheckpoint(dirpath=os.path.join(models_dir, sub_dir), save_top_k=1, monitor="cindex_ipcw", mode='max', filename=filename)
checkpoint_callback3 = ModelCheckpoint(dirpath=os.path.join(models_dir, sub_dir), save_top_k=1, monitor="brier_score", mode='min', filename=filename)
checkpoint_callback = ModelCheckpoint(dirpath=os.path.join(models_dir, sub_dir), save_top_k=1, monitor="all_score", mode='max', filename=filename)

wandb_project = config['run_config']['wandb_project']
wandb_name = f"{date_str}-{config['run_config']['model_type']}-{config['run_config']['model_run']}-{config['run_config']['name']}-{outcome_cols[0]}"
os.makedirs(os.path.join(models_dir, sub_dir), exist_ok=True)

wandb_logger = WandbLogger(project=f"{wandb_project}", offline=False, name=wandb_name, save_dir=os.path.join(models_dir, sub_dir))
wandb_logger.log_hyperparams({**dict(encoder_model.hparams), **config})

callbacks = [checkpoint_callback, checkpoint_callback1, checkpoint_callback2, checkpoint_callback3]

if __name__ == "__main__":
    pl.seed_everything(config['run_config']['random_state'])

    if Path(os.path.join(models_dir, f'{dataset_filename}-train_samples.csv.gz')).exists():
        train_ds_sample_df = pd.read_csv(os.path.join(models_dir, f'{dataset_filename}-train_samples.csv.gz'))
        train_zarrs = train_ds_sample_df['file'].values.tolist()
    else:
        train_ds_sample_df = None

    train_outcome_df = outcome_data[outcome_data['filepath'].isin(train_zarrs)].reset_index(drop=True)
    age_col = config['dataset'].get('age_col', None)
    age_idx = y_demographic_columns.index(age_col) if age_col in y_demographic_columns else None
    if age_col is not None:
        y_demographic_norm_stats = {age_col: {'mean': train_outcome_df[age_col].mean(), 'std': train_outcome_df[age_col].std()}}
    else:
        y_demographic_norm_stats = {}
    print("Demographic normalization stats:", y_demographic_norm_stats)

    train_ds = SingleOutcomeDataset(zarr_files=train_zarrs,
                                                channels=channels, 
                                                max_seq_len_sec=max_seq_len_sec, 
                                                min_seq_len_sec=min_seq_len_sec,
                                                sample_seq_len_sec=seq_len_sec, 
                                                sample_stride_sec=sample_stride,
                                                y_outcome_df = outcome_data,
                                                sample_df=train_ds_sample_df,
                                                trim_wake_epochs=False,
                                                return_hypnogram_every_sec=30,
                                                hypnogram_padding_mask=-100,
                                                hypnogram_frequency=1,
                                                y_mapping_column='filepath',
                                                y_outcome=outcome_cols, 
                                                y_time_column=time_cols,
                                                y_demographic_columns=y_demographic_columns,
                                                y_demographic_norm_stats=y_demographic_norm_stats,
                                                include_partial_samples=config['dataset']['include_partial_samples'], 
                                                frequency=frequency,
                                                butterworth_filters=ALL_FREQUENCY_FILTERS,
                                                median_filter_kernel_size=None,
                                                voltage_channels=None,
                                                constant_nan_tolerance=config['dataset']['constant_nan_tolerance'],
                                                normalize_signals=config['dataset']['normalize_signals'],
                                                clip_interpolations=None)
    
    if not Path(os.path.join(models_dir, f'{dataset_filename}-train_samples.csv.gz')).exists():
        train_ds_sample_df = train_ds.sample_df
        train_ds_sample_df.to_csv(os.path.join(models_dir, f'{dataset_filename}-train_samples.csv.gz'), index=False, compression='gzip')
    
    if Path(os.path.join(models_dir, f'{dataset_filename}-val_samples.csv.gz')).exists():
        val_ds_sample_df = pd.read_csv(os.path.join(models_dir, f'{dataset_filename}-val_samples.csv.gz'))
        val_zarrs = val_ds_sample_df['file'].values.tolist()
    else:
        val_ds_sample_df = None

    val_ds = SingleOutcomeDataset(zarr_files=val_zarrs, 
                                                channels=channels, 
                                                max_seq_len_sec=max_seq_len_sec, 
                                                min_seq_len_sec=min_seq_len_sec,
                                                sample_seq_len_sec=seq_len_sec, 
                                                sample_stride_sec=sample_stride,
                                                y_outcome_df = outcome_data,
                                                sample_df=val_ds_sample_df,
                                                trim_wake_epochs=False,
                                                return_hypnogram_every_sec=30,
                                                hypnogram_padding_mask=-100,
                                                hypnogram_frequency=1,
                                                y_mapping_column='filepath',
                                                y_outcome=outcome_cols, 
                                                y_time_column=time_cols,
                                                y_demographic_columns=y_demographic_columns,
                                                y_demographic_norm_stats=y_demographic_norm_stats,
                                                include_partial_samples=config['dataset']['include_partial_samples'], 
                                                frequency=frequency,
                                                butterworth_filters=ALL_FREQUENCY_FILTERS,
                                                median_filter_kernel_size=None,
                                                voltage_channels=None,
                                                constant_nan_tolerance=config['dataset']['constant_nan_tolerance'],
                                                normalize_signals=config['dataset']['normalize_signals'],
                                                clip_interpolations=None)
    
    if not Path(os.path.join(models_dir, f'{dataset_filename}-val_samples.csv.gz')).exists():
        val_ds_sample_df = val_ds.sample_df
        val_ds_sample_df.to_csv(os.path.join(models_dir, f'{dataset_filename}-val_samples.csv.gz'), index=False, compression='gzip')

    if Path(os.path.join(models_dir, f'{dataset_filename}-test_samples.csv.gz')).exists():
        test_ds_sample_df = pd.read_csv(os.path.join(models_dir, f'{dataset_filename}-test_samples.csv.gz'))
        test_zarrs = test_ds_sample_df['file'].values.tolist()
    else:
        test_ds_sample_df = None

    test_ds = SingleOutcomeDataset(zarr_files=test_zarrs, 
                                                channels=channels, 
                                                max_seq_len_sec=max_seq_len_sec, 
                                                min_seq_len_sec=min_seq_len_sec,
                                                sample_seq_len_sec=seq_len_sec, 
                                                sample_stride_sec=sample_stride,
                                                y_outcome_df = outcome_data,
                                                sample_df=test_ds_sample_df,
                                                trim_wake_epochs=False,
                                                return_hypnogram_every_sec=30,
                                                hypnogram_padding_mask=-100,
                                                hypnogram_frequency=1,
                                                y_mapping_column='filepath',
                                                y_outcome=outcome_cols, 
                                                y_time_column=time_cols,
                                                y_demographic_columns=y_demographic_columns,
                                                y_demographic_norm_stats=y_demographic_norm_stats,
                                                include_partial_samples=config['dataset']['include_partial_samples'], 
                                                frequency=frequency,
                                                butterworth_filters=ALL_FREQUENCY_FILTERS,
                                                median_filter_kernel_size=None,
                                                voltage_channels=None,
                                                constant_nan_tolerance=config['dataset']['constant_nan_tolerance'],
                                                normalize_signals=config['dataset']['normalize_signals'],
                                                clip_interpolations=None)
    
    if not Path(os.path.join(models_dir, f'{dataset_filename}-test_samples.csv.gz')).exists():
        test_ds_sample_df = test_ds.sample_df
        test_ds_sample_df.to_csv(os.path.join(models_dir, f'{dataset_filename}-test_samples.csv.gz'), index=False, compression='gzip')

    if len(ext_test_dfs) > 0:
        if Path(os.path.join(models_dir, f'{dataset_filename}-ext-test_samples.csv.gz')).exists():
            ext_test_ds_sample_df = pd.read_csv(os.path.join(models_dir, f'{dataset_filename}-ext-test_samples.csv.gz'))
            ext_test_zarrs = ext_test_ds_sample_df['file'].values.tolist()
        else:
            ext_test_ds_sample_df = None
            ext_test_zarrs = ext_outcome_df.filepath.unique().tolist()

        ext_test_ds = SingleOutcomeDataset(zarr_files=ext_test_zarrs, 
                                                    channels=channels, 
                                                    max_seq_len_sec=max_seq_len_sec, 
                                                    min_seq_len_sec=min_seq_len_sec,
                                                    sample_seq_len_sec=seq_len_sec, 
                                                    sample_stride_sec=sample_stride,
                                                    y_outcome_df = ext_outcome_df,
                                                    sample_df=ext_test_ds_sample_df,
                                                    trim_wake_epochs=False,
                                                    return_hypnogram_every_sec=30,
                                                    hypnogram_padding_mask=-100,
                                                    hypnogram_frequency=1,
                                                    y_mapping_column='filepath',
                                                    y_outcome=outcome_cols, 
                                                    y_time_column=time_cols,
                                                    y_demographic_columns=y_demographic_columns,
                                                    y_demographic_norm_stats=y_demographic_norm_stats,
                                                    include_partial_samples=config['dataset']['include_partial_samples'], 
                                                    frequency=frequency,
                                                    butterworth_filters=ALL_FREQUENCY_FILTERS,
                                                    median_filter_kernel_size=None,
                                                    voltage_channels=None,
                                                    constant_nan_tolerance=config['dataset']['constant_nan_tolerance'],
                                                    normalize_signals=config['dataset']['normalize_signals'],
                                                    clip_interpolations=None)
        
        if not Path(os.path.join(models_dir, f'{dataset_filename}-ext-test_samples.csv.gz')).exists():
            ext_test_ds_sample_df = ext_test_ds.sample_df
            ext_test_ds_sample_df.to_csv(os.path.join(models_dir, f'{dataset_filename}-ext-test_samples.csv.gz'), index=False, compression='gzip')
        ext_test_loader = DataLoader(ext_test_ds, batch_size=config['training']['batch_size'], shuffle=False, drop_last=False, num_workers=config['training']['num_workers'], persistent_workers=True, pin_memory=False, collate_fn=nested_tensor_sequence_multi_label_collate)
    
    train_loader = DataLoader(train_ds, batch_size=config['training']['batch_size'], shuffle=True, drop_last=True, num_workers=config['training']['num_workers'], persistent_workers=True, pin_memory=False, collate_fn=nested_tensor_sequence_multi_label_collate)
    val_loader = DataLoader(val_ds, batch_size=config['training']['batch_size'], shuffle=False, drop_last=False, num_workers=config['training']['num_workers'], persistent_workers=True, pin_memory=False, collate_fn=nested_tensor_sequence_multi_label_collate)
    test_loader = DataLoader(test_ds, batch_size=config['training']['batch_size'], shuffle=False, drop_last=False, num_workers=config['training']['num_workers'], persistent_workers=True, pin_memory=False, collate_fn=nested_tensor_sequence_multi_label_collate)

    patchmeup_model = PatchTFTSurvivalDemo(learning_rate=config['optimizer']['learning_rate'],
                                        train_size=len(train_ds),
                                        batch_size=config['training']['batch_size'],
                                        n_gpus=config['training']['n_gpus'],
                                        demographic_embeddings=demographic_embeddings,
                                        linear_probing_head=lp_model,
                                        preloaded_model=encoder_model,
                                        discretize_time=True,
                                        discrete_time_bins=config['training']['discrete_time_bins'],
                                        time_range_years=config['training']['time_range_years'],
                                        evaluate_risk_years=config['training']['evaluate_risk_years'],
                                        fine_tune=config['training']['fine_tune'],
                                        epochs=config['training']['epochs'],
                                        scheduler_type=config['scheduler']['scheduler_type'],
                                        optimizer_type=config['optimizer']['optimizer_type'],
                                        weight_decay=config['optimizer']['weight_decay'],
                                        final_weight_decay=config['optimizer']['final_weight_decay'],
                                        use_weight_decay_scheduler=config['optimizer']['use_weight_decay_scheduler'],
                                        scheduler_kwargs=scheduler_kwargs,
                                        age_mlp_hidden_size=config['dataset'].get('age_mlp_hidden_size', 32),
                                        transforms = None
                                        )
                                                  

    trainer = pl.Trainer(precision=config['run_config']['precision'],
                  enable_checkpointing=True,
                  enable_progress_bar=True,
                  enable_model_summary=True,
                  logger=wandb_logger,
                  val_check_interval=config['training']['val_check_interval'],
                  sync_batchnorm=True,
                  strategy="ddp",
                  log_every_n_steps=50,
                  gradient_clip_val=config['training']['gradient_clip_val'],
                  gradient_clip_algorithm='norm' if config['training']['use_gradient_clipping'] else None,
                  num_sanity_val_steps=2,
                  detect_anomaly=False,
                  profiler=None,
                  accelerator="gpu", 
                  accumulate_grad_batches=config['training']['accumulate_grad_batches'],
                  devices=config['training']['n_gpus'],
                  default_root_dir=os.path.join(models_dir, sub_dir), 
                  max_epochs=config['training']['epochs'], 
                  fast_dev_run=False,
                  callbacks=callbacks)
    
    if config['run_config']['do_train']:
        trainer.fit(model=patchmeup_model, train_dataloaders=train_loader, val_dataloaders=val_loader)
        best_model_path = checkpoint_callback.best_model_path
    else:
        print("Loading model from:", best_model_path)
        assert os.path.exists(best_model_path), "Provide path to trained model checkpoint for testing"
    patchmeup_model = PatchTFTSurvivalDemo.load_from_checkpoint(best_model_path, 
                                                                    preloaded_model=encoder_model,
                                                                    linear_probing_head=lp_model,
                                                                    map_location='cpu')
    del train_loader
    train_inference_loader = DataLoader(train_ds, batch_size=config['training']['batch_size'], shuffle=False, drop_last=False, num_workers=config['training']['num_workers'], persistent_workers=True, pin_memory=False, collate_fn=nested_tensor_sequence_multi_label_collate)

    train_preds,train_targets,train_times,train_demographics = get_predictions_nested_survival(data_loader=train_inference_loader, model=patchmeup_model, dataloader_name='train')
    val_preds,val_targets,val_times,val_demographics = get_predictions_nested_survival(data_loader=val_loader, model=patchmeup_model, dataloader_name='val')
    test_preds,test_targets,test_times,test_demographics = get_predictions_nested_survival(data_loader=test_loader, model=patchmeup_model, dataloader_name='test')
    if len(ext_test_dfs) > 0:
        ext_test_preds,ext_test_targets,ext_test_times,ext_test_demographics = get_predictions_nested_survival(data_loader=ext_test_loader, model=patchmeup_model, dataloader_name='ext_test')
        ext_test_preds_cat = torch.cat(ext_test_preds).cpu()
        ext_test_targets_cat = torch.cat(ext_test_targets).cpu()
        ext_test_times_cat = torch.cat(ext_test_times).cpu()
        ext_test_demographics_cat = torch.cat(ext_test_demographics).cpu()
        if age_col is not None:
            ext_test_demographics_cat[:, age_idx] = (ext_test_demographics_cat[:, age_idx] * y_demographic_norm_stats[age_col]['std'] + y_demographic_norm_stats[age_col]['mean'])
    
    train_targets_cat = torch.cat(train_targets).cpu()
    train_preds_cat = torch.cat(train_preds).cpu()
    train_times_cat = torch.cat(train_times).cpu()
    train_demographics_cat = torch.cat(train_demographics).cpu()

    test_targets_cat = torch.cat(test_targets).cpu()
    test_preds_cat = torch.cat(test_preds).cpu()
    test_times_cat = torch.cat(test_times).cpu()
    test_demographics_cat = torch.cat(test_demographics).cpu()

    val_preds_cat = torch.cat(val_preds).cpu()
    val_targets_cat = torch.cat(val_targets).cpu()
    val_times_cat = torch.cat(val_times).cpu()
    val_demographics_cat = torch.cat(val_demographics).cpu()

    if age_col is not None:
        train_demographics_cat[:, age_idx] = (train_demographics_cat[:, age_idx] * y_demographic_norm_stats[age_col]['std'] + y_demographic_norm_stats[age_col]['mean'])
        val_demographics_cat[:, age_idx] = (val_demographics_cat[:, age_idx] * y_demographic_norm_stats[age_col]['std'] + y_demographic_norm_stats[age_col]['mean'])
        test_demographics_cat[:, age_idx] = (test_demographics_cat[:, age_idx] * y_demographic_norm_stats[age_col]['std'] + y_demographic_norm_stats[age_col]['mean'])
    
    if len(ext_test_dfs) > 0:
        tensor_dict = {'train_targets': train_targets_cat, 'train_preds': train_preds_cat,'train_times':train_times_cat, 'train_demographics':train_demographics_cat,
                    'val_targets': val_targets_cat, 'val_preds': val_preds_cat,'val_times':val_times_cat, 'val_demographics':val_demographics_cat,
                    'test_targets': test_targets_cat, 'test_preds': test_preds_cat,'test_times':test_times_cat, 'test_demographics':test_demographics_cat,
                    'ext_test_targets':ext_test_targets_cat, 'ext_test_preds': ext_test_preds_cat, 'ext_test_times':ext_test_times_cat, 'ext_test_demographics':ext_test_demographics_cat
                    }
    else:
        tensor_dict = {'train_targets': train_targets_cat, 'train_preds': train_preds_cat,'train_times':train_times_cat, 'train_demographics':train_demographics_cat,
                    'val_targets': val_targets_cat, 'val_preds': val_preds_cat,'val_times':val_times_cat, 'val_demographics':val_demographics_cat,
                    'test_targets': test_targets_cat, 'test_preds': test_preds_cat,'test_times':test_times_cat, 'test_demographics':test_demographics_cat
                    }
    if config['run_config']['do_train']:
        torch.save(tensor_dict, os.path.join(models_dir, sub_dir, f'{"_".join(outcome_cols)}-{config["run_config"]["name"]}-predictions.pt'))
    else:
        dir_ = os.path.dirname(best_model_path)
        torch.save(tensor_dict, os.path.join(dir_, f'{"_".join(outcome_cols)}-{config["run_config"]["name"]}-predictions.pt'))
