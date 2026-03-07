import torch
import pandas as pd
import os
import glob
import lightning.pytorch as pl
import yaml
from sleepjepa.jepa import JEPASimpleLightning, jepa_mse_loss

from sleepjepa.slumber import SelfSupervisedHypnogramTimeDataset, nested_tensor_sequence_collate, padded_tensor_sequence_collate, ALL_FREQUENCY_FILTERS
from sleepjepa.augmentations import random_crop, jitter_augmentation, TransformsCallback
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader
from datetime import datetime
import zarr
from zclip import ZClipLightningCallback

zclip_cb = ZClipLightningCallback(mode="zscore", alpha=0.97, z_thresh=2.5, clip_option="adaptive_scaling", max_grad_norm=10.0, clip_factor=1.0)

zarr.config.set({'async.concurrency': 512})


from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.tuner import Tuner

from torch import nn

from pathlib import Path
from functools import partial
from sleepjepa.slumber import ALL_CHANNELS

torch.set_float32_matmul_precision('high')
torch.backends.cuda.enable_flash_sdp(True)

config_path = 'train_patch_jepa.yaml'
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

test_n_samples = config['run_config']['test_n_samples']
channels = ALL_CHANNELS

loss_fxn = jepa_mse_loss

c_in = len(channels)
max_seq_len_sec = config['dataset']['max_seq_len_sec']
min_seq_len_sec = config['dataset']['min_seq_len_sec']
frequency = config['dataset']['frequency']

patch_seconds = config['dataset']['patch_seconds']
win_length = frequency*patch_seconds
overlap = config['dataset']['overlap']
hop_length=win_length - int(overlap*win_length)
seq_len_sec = config['dataset']['seq_len_sec']
sample_stride_sec = config['dataset']['sample_stride_sec']
max_seq_len = seq_len_sec*frequency

padded_tensor_sequence_collate_partial = partial(padded_tensor_sequence_collate, max_seq_len=max_seq_len, frequency=frequency, return_hypnogram=False, hypnogram_frequency=1, X_pad_value = 0., hypnogram_padding_mask = -100)

n_patches = (max(max_seq_len, win_length)-win_length) // hop_length + 1
if ((max_seq_len-win_length) % hop_length != 0):
    n_patches += 1
n_patches = int(n_patches)

use_transforms = config['dataset']['use_transforms']
if use_transforms:
    transforms_callback = TransformsCallback(
        transforms=[
            partial(random_crop, c_in=c_in, min_len=(min_seq_len_sec)*frequency, p=0.5),
            partial(jitter_augmentation, mask_ratio=0.05, jitter_ratio=0.05, p=0.5)
        ]
    )
else:
    transforms_callback = None

train_ds_file_path = config['paths']['train_ds_file_path']
val_ds_file_path = config['paths']['val_ds_file_path']

models_dir = config['paths']['models_dir']
data_dir = config['paths']['data_dir']

trim_wake_epochs = config['dataset']['trim_wake_epochs']
return_hypnogram_every_sec = config['dataset']['return_hypnogram_every_sec']
hypnogram_frequency = config['dataset']['hypnogram_frequency']
hypnogram_padding_mask = config['dataset']['hypnogram_padding_mask']

precision = config['run_config']['precision']

val_check_interval=config['training']['val_check_interval']

use_gradient_clipping = config['training']['use_gradient_clipping']
gradient_clip_val = config['training']['gradient_clip_val']
gradient_clip_algorithm = config['training']['gradient_clip_algorithm']

scheduler_kwargs = config['scheduler'].copy()
del scheduler_kwargs['scheduler_type']

learning_rate = config['optimizer']['learning_rate']
weight_decay=config['optimizer']['weight_decay']
use_weight_decay_scheduler=config['optimizer']['use_weight_decay_scheduler']
final_weight_decay=config['optimizer']['final_weight_decay']

BATCHSIZE = config['training']['batch_size']
accumulate_grad_batches = config['training']['accumulate_grad_batches']
EPOCHS = config['training']['epochs']
n_gpus = config['training']['n_gpus']
num_workers = config['training']['num_workers']
huber_delta = config['training']['huber_delta']
include_partial_samples = config['dataset']['include_partial_samples']
random_state=config['run_config']['random_state']

start_offset_sec=config['dataset']['start_offset_sec']

zarr_files_hsp = glob.glob(os.path.join(data_dir, "human_sleep_project_waveforms_no_resampling/*.zarr/"))
aim_ahi_files = glob.glob(os.path.join(data_dir, "aim_ahi_waveforms_no_resampling_fix/*.zarr/"))
apples_zarr_files = glob.glob(os.path.join(data_dir, "apples_waveforms_no_resampling/*.zarr/"))
wsc_zarr_files = glob.glob(os.path.join(data_dir, "wsc_waveforms_no_resampling/*.zarr/"))
mnc_zarr_files = glob.glob(os.path.join(data_dir, "mnc_waveforms_no_resampling/*.zarr/"))
groups_hsp = [Path(i).stem.split('_ses')[0] for i in zarr_files_hsp]
groups_wsc = [Path(i).stem.split('-')[-1] for i in wsc_zarr_files]
groups_apples = [Path(i).stem.split('-')[-1] for i in apples_zarr_files]
groups_aim_ahi = [Path(i).stem.split('_')[-1] for i in aim_ahi_files]
groups_mnc = [Path(i).stem.split('-nsrr')[0] for i in mnc_zarr_files]

zarr_files_pretrain = zarr_files_hsp + apples_zarr_files + wsc_zarr_files + aim_ahi_files + mnc_zarr_files
groups_pretrain = groups_hsp + groups_apples + groups_wsc + groups_aim_ahi + groups_mnc

TRAIN_SIZE_PRETRAIN = 0.97
splitter = GroupShuffleSplit(n_splits=1, train_size=TRAIN_SIZE_PRETRAIN, random_state=random_state)
train_idxs_pretrain, valid_idxs_pretrain = next(splitter.split(X=zarr_files_pretrain, groups=groups_pretrain))

train_zarrs = [zarr_files_pretrain[i] for i in train_idxs_pretrain]
val_zarrs = [zarr_files_pretrain[i] for i in valid_idxs_pretrain]

if config['training']['linear_probe']:
    val_zarrs = [i for i in val_zarrs if Path(f'{i}/hypnogram').exists() and 'human_sleep_project' not in i]



encoder_arch = dict(c_in=c_in,
            num_patches=n_patches,
            patch_size=win_length,
            patch_stride=hop_length,
            d_model=config['encoder']['d_model'],
            nhead=config['encoder']['nhead'],
            use_tst_block=config['encoder']['use_tst_block'],
            shared_embedding=config['encoder']['shared_embedding'],
            num_layers=config['encoder']['num_layers'],
            pe_type=config['encoder']['pe_type'],
            mlp_ratio=config['encoder']['mlp_ratio'],
            qkv_bias=config['encoder']['qkv_bias'],
            qk_scale=config['encoder']['qk_scale'],
            drop_rate=config['encoder']['drop_rate'],
            attn_drop_rate=config['encoder']['attn_drop_rate'],
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            jepa=config['encoder']['jepa'],
            tokenizer_type=config['encoder']['tokenizer_type'],
            tokenizer_kwargs=config['encoder']['tokenizer_kwargs'],
            embed_activation=nn.GELU())

predictor_arch = dict(num_patches=n_patches,
            encoder_embed_dim=encoder_arch['d_model'],
            predictor_embed_dim=config['predictor']['predictor_embed_dim'],
            nhead=config['predictor']['nhead'],
            num_layers=config['predictor']['num_layers'],
            pe_type=config['predictor']['pe_type'],
            mlp_ratio=config['predictor']['mlp_ratio'],
            qkv_bias=config['predictor']['qkv_bias'],
            qk_scale=config['predictor']['qk_scale'],
            drop_rate=config['predictor']['drop_rate'],
            attn_drop_rate=config['predictor']['attn_drop_rate'],
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            use_tst_block=config['predictor']['use_tst_block'],
            c_in_mask_tokens=config['predictor']['c_in_mask_tokens'],
            shuffle=config['predictor'].get('shuffle', True),
            embed_activation=nn.GELU())


name = config['run_config']['name']
filename = f"{name}" + "{epoch:02d}-{train_loss:.5f}-{val_loss:.5f}-{val_auroc:.5f}"
date_str = datetime.now().strftime("%Y-%m-%d")
sub_dir = f"{date_str}-{config['run_config']['model_type']}-{config['run_config']['model_run']}" if config['run_config']['checkpoint_start_train_path'] is None else Path(config['run_config']['checkpoint_start_train_path']).parent.stem
checkpoint_callback = ModelCheckpoint(dirpath=os.path.join(models_dir, sub_dir), save_top_k=EPOCHS, monitor="train_loss", mode='min', filename=filename, save_last=False, save_on_train_epoch_end=True)
wandb_project = config['run_config']['wandb_project']
wandb_name = f"{date_str}-{config['run_config']['model_type']}-{config['run_config']['model_run']}-{config['run_config']['name']}"
os.makedirs(os.path.join(models_dir, sub_dir), exist_ok=True)

wandb_logger = WandbLogger(project=f'{wandb_project}', offline=False, name=wandb_name, save_dir=os.path.join(models_dir, sub_dir))
wandb_logger.log_hyperparams(config)
if __name__ == "__main__":
    pl.seed_everything(random_state)
    if Path(train_ds_file_path).exists():
        train_ds_sample_df = pd.read_csv(train_ds_file_path, low_memory=False)
        train_zarrs = train_ds_sample_df['file'].values.tolist()
    else:
        train_ds_sample_df = None
    if test_n_samples > 0:
        train_ds_sample_df = train_ds_sample_df.sample(n=test_n_samples)
        train_zarrs = train_ds_sample_df['file'].values.tolist()
    train_ds = SelfSupervisedHypnogramTimeDataset(zarr_files=train_zarrs,
                                            channels=channels, 
                                            frequency=frequency,
                                            trim_wake_epochs=trim_wake_epochs,
                                            sample_df=train_ds_sample_df,
                                            return_hypnogram_every_sec=return_hypnogram_every_sec,
                                            hypnogram_frequency=hypnogram_frequency,
                                            hypnogram_padding_mask=hypnogram_padding_mask,
                                            start_offset_sec=start_offset_sec,
                                            clip_interpolations=None,
                                            include_partial_samples=include_partial_samples, 
                                            butterworth_filters=ALL_FREQUENCY_FILTERS,
                                            normalize_signals=config['dataset']['normalize_signals'],
                                            median_filter_kernel_size=None,
                                            voltage_channels=None,
                                            max_seq_len_sec=max_seq_len_sec, 
                                            min_seq_len_sec=min_seq_len_sec,
                                            sample_seq_len_sec=seq_len_sec, 
                                            sample_stride_sec=sample_stride_sec,
                                            constant_nan_tolerance=config['dataset']['constant_nan_tolerance'],
                                            return_hyponogram=False)
    if not Path(train_ds_file_path).exists():
        train_ds_sample_df = train_ds.sample_df
        train_ds_sample_df.to_csv(train_ds_file_path, index=False, compression='gzip')

    if Path(val_ds_file_path).exists():
        val_ds_sample_df = pd.read_csv(val_ds_file_path, low_memory=False)
        val_zarrs = val_ds_sample_df['file'].values.tolist()
    else:
        val_ds_sample_df = None

    if test_n_samples > 0:
        val_ds_sample_df = val_ds_sample_df.sample(n=test_n_samples // 2)
        val_zarrs = val_ds_sample_df['file'].values.tolist()
    val_ds = SelfSupervisedHypnogramTimeDataset(zarr_files=val_zarrs, 
                                            channels=channels, 
                                            frequency=frequency,
                                            sample_df=val_ds_sample_df,
                                            trim_wake_epochs=trim_wake_epochs,
                                            return_hypnogram_every_sec=return_hypnogram_every_sec,
                                            hypnogram_frequency=hypnogram_frequency,
                                            hypnogram_padding_mask=hypnogram_padding_mask,
                                            start_offset_sec=start_offset_sec,
                                            clip_interpolations=None,
                                            include_partial_samples=include_partial_samples, 
                                            butterworth_filters=ALL_FREQUENCY_FILTERS,
                                            normalize_signals=config['dataset']['normalize_signals'],
                                            median_filter_kernel_size=None,
                                            voltage_channels=None,
                                            max_seq_len_sec=max_seq_len_sec, 
                                            min_seq_len_sec=min_seq_len_sec,
                                            sample_seq_len_sec=seq_len_sec, 
                                            sample_stride_sec=sample_stride_sec,
                                            constant_nan_tolerance=config['dataset']['constant_nan_tolerance'],
                                            return_hyponogram=False if not config['training']['linear_probe'] else True)
    if not Path(val_ds_file_path).exists():
        val_ds_sample_df = val_ds.sample_df
        val_ds_sample_df.to_csv(val_ds_file_path, index=False, compression='gzip')
    train_loader = DataLoader(train_ds, batch_size=BATCHSIZE, shuffle=True, num_workers=num_workers, drop_last=True, persistent_workers=False, pin_memory=False if not config['dataset']['nested'] else False, collate_fn=nested_tensor_sequence_collate if config['dataset']['nested'] else padded_tensor_sequence_collate_partial)
    val_loader = DataLoader(val_ds, batch_size=BATCHSIZE, shuffle=False, num_workers=num_workers, drop_last=False, persistent_workers=False, pin_memory=False if not config['dataset']['nested'] else False, collate_fn=nested_tensor_sequence_collate if config['dataset']['nested'] else padded_tensor_sequence_collate_partial)
    patchfreq_model = JEPASimpleLightning(learning_rate=learning_rate,
                                        train_size=len(train_ds),
                                        batch_size=BATCHSIZE,
                                        n_gpus=n_gpus,
                                        num_nodes=config['training']['num_nodes'],
                                        patchtsjepa_encoder_kwargs=encoder_arch,
                                        patchtsjepa_predictor_kwargs=predictor_arch,
                                        weight_decay=weight_decay,
                                        use_weight_decay_scheduler=use_weight_decay_scheduler,
                                        final_weight_decay=final_weight_decay,
                                        epochs=EPOCHS,
                                        loss_fn=loss_fxn,
                                        optimizer_type=config['optimizer']['optimizer_type'],
                                        scheduler_type=config['scheduler']['scheduler_type'],
                                        target_mask_range=config['training']['target_mask_range'],
                                        context_mask_range=config['training']['context_mask_range'],
                                        mask_block_range=config['training']['mask_block_range'],
                                        ema_decay=config['training']['ema_decay'],
                                        scheduler_kwargs=scheduler_kwargs,
                                        transforms=transforms_callback,
                                        linear_probe=config['training']['linear_probe'],
                                        )
    
    wandb_logger.watch(patchfreq_model, log="all", log_graph=False)
    
    trainer = pl.Trainer(precision=precision,
                     enable_checkpointing=True,
                     enable_progress_bar=True,
                     enable_model_summary=True,
                     logger=wandb_logger, 
                     strategy="ddp",
                     sync_batchnorm=True,
                     val_check_interval=val_check_interval,
                     check_val_every_n_epoch=config['training'].get('check_val_every_n_epoch', 1),
                     gradient_clip_val=gradient_clip_val,
                     gradient_clip_algorithm=gradient_clip_algorithm if use_gradient_clipping else None,
                     num_sanity_val_steps=2,
                     detect_anomaly=False,
                     accelerator="gpu", 
                     accumulate_grad_batches=accumulate_grad_batches,
                     devices=n_gpus, 
                     num_nodes=config['training']['num_nodes'],
                     default_root_dir=os.path.join(models_dir, sub_dir), 
                     max_epochs=EPOCHS, 
                     fast_dev_run=False,
                     callbacks=[zclip_cb, checkpoint_callback])
    
    if config['training']['use_lr_finder']:
        tuner = Tuner(trainer)
        lr_finder = tuner.lr_find(patchfreq_model, train_dataloaders=train_loader, update_attr=False, attr_name="max_lr")
        new_lr = lr_finder.suggestion()
        patchfreq_model.scheduler_kwargs['max_lr'] = new_lr
        print(f"Using max lr: {new_lr}")
    
    trainer.fit(model=patchfreq_model, train_dataloaders=train_loader, val_dataloaders=val_loader, ckpt_path=config['run_config']['checkpoint_start_train_path'])
