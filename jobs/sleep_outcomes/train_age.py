import torch
import pandas as pd
import os
import lightning.pytorch as pl
import yaml
from sleepjepa.jepa import JEPASimpleLightning, LeJEPALightning
from sleepjepa.train import PatchTFTSingleOutcomeLightning
from sleepjepa.nested import get_predictions_nested

from sleepjepa.slumber import SingleOutcomeDataset, ALL_CHANNELS, nested_tensor_collate, ALL_FREQUENCY_FILTERS
from lightning.pytorch.tuner import Tuner
from datetime import datetime
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader

from sleepjepa.heads import AttentiveClassifier
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from pathlib import Path
import torch.nn as nn
from functools import partial
from sleepjepa.augmentations import TransformsCallback, jitter_augmentation

metrics = {}

torch.backends.cuda.enable_flash_sdp(True)
torch.set_float32_matmul_precision('high')

config_path = 'config/train_age.yaml'
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

use_transforms = config['training']['use_transforms']
if use_transforms:
    transforms_callback = TransformsCallback(
        transforms=[
            partial(jitter_augmentation, mask_ratio=0.05, jitter_ratio=0.05, p=0.5),
        ]
    )
else:
    transforms_callback = None

precision = config['run_config']['precision']
encoder_dir = config['paths']['encoder_dir']
models_dir = config['paths']['models_dir']
dataset_filename = config['paths']['dataset_filename']

pretrained_encoder_path = os.path.join(encoder_dir, config['paths']['pretrained_encoder_path'])

try:
    encoder_model = JEPASimpleLightning.load_from_checkpoint(pretrained_encoder_path, map_location='cpu')
    num_heads = encoder_model.encoder.predictor_blocks[0].self_attn.num_heads
    d_model = encoder_model.encoder.d_model
except:
    encoder_model = LeJEPALightning.load_from_checkpoint(pretrained_encoder_path, map_location='cpu')
    num_heads = encoder_model.encoder.encoder.predictor_blocks[0].self_attn.num_heads
    d_model = encoder_model.encoder.encoder.d_model

BATCHSIZE = config['training']['batch_size']
patch_seconds = config['dataset']['patch_seconds']

model_run = config['run_config']['model_run']
name = config['run_config']['name']

val_check_interval = config['training']['val_check_interval']

scheduler_kwargs = config['scheduler'].copy()
scheduler_type = scheduler_kwargs['scheduler_type']
del scheduler_kwargs['scheduler_type']

learning_rate = config['optimizer']['learning_rate']
weight_decay=config['optimizer']['weight_decay']
use_weight_decay_scheduler=config['optimizer']['use_weight_decay_scheduler']
final_weight_decay=config['optimizer']['final_weight_decay']

gradient_clip_val = config['training']['gradient_clip_val']
use_gradient_clipping=config['training']['use_gradient_clipping']

scheduler_type = config['scheduler']['scheduler_type']

outcome_cols = config['dataset']['outcome_cols']
n_labels = len(outcome_cols)

accumulate_grad_batches = config['training']['accumulate_grad_batches']
EPOCHS = config['training']['epochs']
n_gpus = config['training']['n_gpus']
num_workers = config['training']['num_workers']

fine_tune = config['training']['fine_tune']

random_state=config['run_config']['random_state']

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
splitter = GroupShuffleSplit(n_splits=1, train_size=0.8, random_state=random_state)

groups = [Path(i).stem for i in outcome_data.filepath.unique()]
train_idxs, test_idxs = next(splitter.split(X=outcome_data.filepath.unique(), groups=groups))
train_val_zarrs = [outcome_data.filepath.unique()[i] for i in train_idxs]
test_zarrs.extend([outcome_data.filepath.unique()[i] for i in test_idxs])

groups = [Path(i).stem for i in train_val_zarrs]
splitter2 = GroupShuffleSplit(n_splits=1, train_size=0.9, random_state=random_state)
train_idxs, val_idxs = next(splitter2.split(X=train_val_zarrs, groups=groups))
train_zarrs.extend([train_val_zarrs[i] for i in train_idxs])
val_zarrs.extend([train_val_zarrs[i] for i in val_idxs])

print(f"Number of training samples: {len(train_zarrs)}")
print(f"Number of validation samples: {len(val_zarrs)}")
print(f"Number of test samples: {len(test_zarrs)}")
if len(ext_test_dfs) > 0:
    print(f"Number of external test samples: {len(ext_outcome_df.filepath.unique().tolist())}")

channels = ALL_CHANNELS 
c_in = len(channels)

frequency = config['dataset']['frequency']

win_length=frequency*patch_seconds 
overlap = config['dataset']['overlap']
hop_length=win_length - int(overlap*win_length)
max_seq_len_sec = config['dataset']['max_seq_len_sec']
seq_len_sec = sample_stride = max_seq_len_sec
max_seq_len = seq_len_sec*frequency if seq_len_sec is not None else max_seq_len_sec*frequency
include_partial_samples = config['dataset']['include_partial_samples']
n_patches = (max(max_seq_len, win_length)-win_length) // hop_length + 1
if ((max_seq_len-win_length) % hop_length != 0):
    n_patches += 1

lp_head = dict(embed_dim=d_model,
        num_heads=config['lp_head'].get('num_heads', num_heads),
        mlp_ratio=config['lp_head']['mlp_ratio'],
        depth=config['lp_head']['depth'],
        c_in=c_in,
        norm_layer=nn.LayerNorm,
        init_std=config['lp_head']['init_std'],
        qkv_bias=config['lp_head']['qkv_bias'],
        num_classes=1,
        complete_block=config['lp_head']['complete_block'],
        per_channel=config['lp_head'].get('per_channel', False)
    )
lp_model = AttentiveClassifier(**lp_head)

name = config['run_config']['name']
filename = f"{name}-{outcome_cols[0]}" + "{epoch:02d}-Loss-{val_loss:.5f}"
date_str = datetime.now().strftime("%Y-%m-%d")
sub_dir = f"{date_str}-{config['run_config']['model_type']}-{config['run_config']['model_run']}-{outcome_cols[0]}" if config['run_config']['do_train'] else Path(config['run_config']['test_checkpoint_path']).parent.stem

checkpoint_callback = ModelCheckpoint(dirpath=os.path.join(models_dir, sub_dir), save_top_k=1, monitor="val_loss", mode='min', filename=filename)

wandb_project = config['run_config']['wandb_project']
wandb_name = f"{date_str}-{config['run_config']['model_type']}-{config['run_config']['model_run']}-{config['run_config']['name']}-{outcome_cols[0]}"
os.makedirs(os.path.join(models_dir, sub_dir), exist_ok=True)

wandb_logger = WandbLogger(project=f"{wandb_project}", offline=False, name=wandb_name, save_dir=os.path.join(models_dir, sub_dir))
wandb_logger.log_hyperparams({**dict(encoder_model.hparams), **config})

callbacks = [checkpoint_callback]

if __name__ == "__main__":
    pl.seed_everything(random_state)

    if Path(os.path.join(models_dir, f'{dataset_filename}-train_samples.csv.gz')).exists():
        train_ds_sample_df = pd.read_csv(os.path.join(models_dir, f'{dataset_filename}-train_samples.csv.gz'))
        train_zarrs = train_ds_sample_df['file'].values.tolist()
    else:
        train_ds_sample_df = None

    train_ds = SingleOutcomeDataset(zarr_files=train_zarrs,
                                                channels=channels, 
                                                max_seq_len_sec=max_seq_len_sec, 
                                                min_seq_len_sec=config['dataset']['min_seq_len_sec'],
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
                                                y_time_column=None,
                                                include_partial_samples=include_partial_samples, 
                                                frequency=frequency,
                                                constant_nan_tolerance=config['dataset']['constant_nan_tolerance'],
                                                butterworth_filters=ALL_FREQUENCY_FILTERS,
                                                median_filter_kernel_size=None,
                                                voltage_channels=None,
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
                                                min_seq_len_sec=config['dataset']['min_seq_len_sec'],
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
                                                y_time_column=None,
                                                include_partial_samples=include_partial_samples, 
                                                frequency=frequency,
                                                butterworth_filters=ALL_FREQUENCY_FILTERS,
                                                median_filter_kernel_size=None,
                                                constant_nan_tolerance=config['dataset']['constant_nan_tolerance'],
                                                voltage_channels=None,
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
                                                min_seq_len_sec=config['dataset']['min_seq_len_sec'],
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
                                                y_time_column=None,
                                                include_partial_samples=include_partial_samples, 
                                                frequency=frequency,
                                                butterworth_filters=ALL_FREQUENCY_FILTERS,
                                                median_filter_kernel_size=None,
                                                constant_nan_tolerance=config['dataset']['constant_nan_tolerance'],
                                                voltage_channels=None,
                                                normalize_signals=config['dataset']['normalize_signals'],
                                                clip_interpolations=None)
    if not Path(os.path.join(models_dir, f'{dataset_filename}-test_samples.csv.gz')).exists():
        test_ds_sample_df = test_ds.sample_df
        test_ds_sample_df.to_csv(os.path.join(models_dir, f'{dataset_filename}-test_samples.csv.gz'), index=False, compression='gzip')

    if Path(os.path.join(models_dir, f'{dataset_filename}-ext-test_samples.csv.gz')).exists():
        ext_test_ds_sample_df = pd.read_csv(os.path.join(models_dir, f'{dataset_filename}-ext-test_samples.csv.gz'))
        ext_test_zarrs = ext_test_ds_sample_df['file'].values.tolist()
    else:
        ext_test_ds_sample_df = None
        ext_test_zarrs = ext_outcome_df.filepath.unique().tolist()
    
    ext_test_ds = SingleOutcomeDataset(zarr_files=ext_test_zarrs, 
                                                channels=channels, 
                                                max_seq_len_sec=max_seq_len_sec, 
                                                min_seq_len_sec=config['dataset']['min_seq_len_sec'],
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
                                                y_time_column=None,
                                                include_partial_samples=include_partial_samples, 
                                                frequency=frequency,
                                                butterworth_filters=ALL_FREQUENCY_FILTERS,
                                                median_filter_kernel_size=None,
                                                constant_nan_tolerance=config['dataset']['constant_nan_tolerance'],
                                                voltage_channels=None,
                                                normalize_signals=config['dataset']['normalize_signals'],
                                                clip_interpolations=None)
    if not Path(os.path.join(models_dir, f'{dataset_filename}-ext-test_samples.csv.gz')).exists():
        ext_test_ds_sample_df = ext_test_ds.sample_df
        ext_test_ds_sample_df.to_csv(os.path.join(models_dir, f'{dataset_filename}-ext-test_samples.csv.gz'), index=False, compression='gzip')
    
    train_loader = DataLoader(train_ds, batch_size=BATCHSIZE, shuffle=True, drop_last=True, num_workers=num_workers, persistent_workers=True, pin_memory=False, collate_fn=nested_tensor_collate)
    val_loader = DataLoader(val_ds, batch_size=BATCHSIZE, shuffle=False, drop_last=False, num_workers=num_workers, persistent_workers=True, pin_memory=False, collate_fn=nested_tensor_collate)
    test_loader = DataLoader(test_ds, batch_size=BATCHSIZE, shuffle=False, drop_last=False, num_workers=num_workers, persistent_workers=True, pin_memory=False, collate_fn=nested_tensor_collate)
    ext_test_loader = DataLoader(ext_test_ds, batch_size=BATCHSIZE, shuffle=False, drop_last=False, num_workers=num_workers, persistent_workers=True, pin_memory=False, collate_fn=nested_tensor_collate)

    patchmeup_model = PatchTFTSingleOutcomeLightning(learning_rate=learning_rate,
                                    train_size=len(train_ds),
                                    n_gpus=n_gpus,
                                    batch_size=BATCHSIZE,
                                    linear_probing_head=lp_model,
                                    preloaded_model=encoder_model,
                                    metrics=metrics,
                                    class_weights=None,
                                    fine_tune=fine_tune,
                                    epochs=EPOCHS,
                                    scheduler_type=scheduler_type,
                                    optimizer_type=config['optimizer']['optimizer_type'],
                                    weight_decay=weight_decay,
                                    use_weight_decay_scheduler=use_weight_decay_scheduler,
                                    final_weight_decay=final_weight_decay,
                                    scheduler_kwargs=scheduler_kwargs,
                                    mixup_callback=None,
                                    transforms=transforms_callback,
                                    regression=True,
                                    loss_func=nn.MSELoss()
                                    )
                                                  

    trainer = pl.Trainer(precision=precision,
                  enable_checkpointing=True,
                  enable_progress_bar=True,
                  enable_model_summary=True,
                  logger=wandb_logger,
                  val_check_interval=val_check_interval,
                  sync_batchnorm=True,
                  strategy="ddp",
                  log_every_n_steps=50,
                  gradient_clip_val=gradient_clip_val,
                  gradient_clip_algorithm='norm' if use_gradient_clipping else None,
                  num_sanity_val_steps=2,
                  detect_anomaly=False,
                  profiler=None,
                  accelerator="gpu", 
                  accumulate_grad_batches=accumulate_grad_batches,
                  devices=n_gpus,
                  default_root_dir=os.path.join(models_dir, sub_dir), 
                  max_epochs=EPOCHS, 
                  fast_dev_run=False,
                  callbacks=callbacks)
    if config['training']['use_lr_finder']:
        tuner = Tuner(trainer)
        lr_finder = tuner.lr_find(patchmeup_model, train_dataloaders=train_loader, update_attr=False, attr_name="max_lr")
        new_lr = lr_finder.suggestion()
        patchmeup_model.scheduler_kwargs['max_lr'] = new_lr
        print(f"Using max lr: {new_lr}")

    if config['run_config']['do_train']:
        trainer.fit(model=patchmeup_model, train_dataloaders=train_loader, val_dataloaders=val_loader)
        best_model_path = checkpoint_callback.best_model_path
    else:
        assert os.path.exists(config['run_config']['test_checkpoint_path']), "Provide path to trained model checkpoint for testing"
        best_model_path = config['run_config']['test_checkpoint_path']
    
    patchmeup_model = PatchTFTSingleOutcomeLightning.load_from_checkpoint(best_model_path, 
                                                                          linear_probing_head=lp_model,
                                                                          preloaded_model=encoder_model,
                                                                          map_location='cpu')
    val_preds,val_targets = get_predictions_nested(data_loader=val_loader, model=patchmeup_model, dataloader_name='val')
    test_preds,test_targets = get_predictions_nested(data_loader=test_loader, model=patchmeup_model, dataloader_name='test')
    ext_test_preds,ext_test_targets = get_predictions_nested(data_loader=ext_test_loader, model=patchmeup_model, dataloader_name='ext_test')
    
    test_targets_cat = torch.cat(test_targets).cpu()
    test_preds_cat = torch.cat(test_preds).cpu()

    val_preds_cat = torch.cat(val_preds).cpu()
    val_targets_cat = torch.cat(val_targets).cpu()
    
    ext_test_preds_cat = torch.cat(ext_test_preds).cpu()
    ext_test_targets_cat = torch.cat(ext_test_targets).cpu()

    tensor_dict = {'val_targets': val_targets_cat, 'val_preds': val_preds_cat,
                   'test_targets': test_targets_cat, 'test_preds': test_preds_cat,
                   'ext_test_targets':ext_test_targets_cat, 'ext_test_preds': ext_test_preds_cat
                   }
    if config['run_config']['do_train']:
        torch.save(tensor_dict, os.path.join(models_dir, sub_dir, f'{"_".join(outcome_cols)}-{config["run_config"]["name"]}-predictions.pt'))
    else:
        dir_ = os.path.dirname(config['run_config']['test_checkpoint_path'])
        torch.save(tensor_dict, os.path.join(dir_, f'{"_".join(outcome_cols)}-{config["run_config"]["name"]}-predictions.pt'))
