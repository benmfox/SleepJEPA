import torch
import pandas as pd
import os
import glob
import lightning.pytorch as pl
import torch.nn.functional as F

from sleepjepa.jepa import JEPASimpleLightning
from sleepjepa.augmentations import VariableChannelInput
from sleepjepa.train import PatchTFTSleepStage
from sleepjepa.nested import get_predictions_nested
from sleepjepa.heads import RNNProbingHead
from sleepjepa.slumber import ALL_CHANNELS, nested_tensor_sequence_collate, SelfSupervisedHypnogramTimeDataset, ALL_FREQUENCY_FILTERS

from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader

from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from datetime import datetime
import yaml
from pathlib import Path
from torchmetrics.classification import MulticlassAUROC, MulticlassAveragePrecision, MulticlassAccuracy
from datetime import datetime




metrics = {'auroc':MulticlassAUROC(num_classes=5, average='macro', ignore_index=-100),
           'ap':MulticlassAveragePrecision(num_classes=5, average='macro', ignore_index=-100),
           'acc':MulticlassAccuracy(num_classes=5, average='macro', ignore_index=-100)}

torch.set_float32_matmul_precision('high')
torch.backends.cuda.enable_flash_sdp(True)

config_path = 'config/train_sleep_stages.yaml'
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

dataset_filename = config['paths']['dataset_filename']

random_state = config['run_config']['random_state']
encoder_dir = config['paths']['encoder_dir']
models_dir = config['paths']['models_dir']
data_dir = config['paths']['data_dir']
loss_fxn = config['run_config']['loss_fxn']
pretrained_encoder_path = config['paths']['pretrained_encoder_path']

if not Path(os.path.join(models_dir, f'{dataset_filename}-train_samples.csv.gz')).exists():
    zarr_files_shhs1 = glob.glob(os.path.join(data_dir, "shhs1_waveforms_no_resampling/*.zarr/hypnogram"))
    zarr_files_shhs1 = list(map(lambda x: str(Path(x).parent), zarr_files_shhs1))
    
    zarr_files_shhs2 = glob.glob(os.path.join(data_dir, "shhs2_waveforms_no_resampling/*.zarr/hypnogram"))
    zarr_files_shhs2 = list(map(lambda x: str(Path(x).parent), zarr_files_shhs2))
    
    
    mesa_zarr_files = glob.glob(os.path.join(data_dir, "mesa_waveforms_no_resampling/*.zarr/hypnogram"))
    mesa_zarr_files = list(map(lambda x: str(Path(x).parent), mesa_zarr_files))
    
    zarr_files_wsc = glob.glob(os.path.join(data_dir, "wsc_waveforms_no_resampling/*.zarr/hypnogram"))
    zarr_files_wsc = list(map(lambda x: str(Path(x).parent), zarr_files_wsc))
    apples_zarr_files = glob.glob(os.path.join(data_dir, "apples_waveforms_no_resampling/*.zarr/hypnogram")) 
    apples_zarr_files = list(map(lambda x: str(Path(x).parent), apples_zarr_files))
    
    zarr_files_mros_visit1 = glob.glob(os.path.join(data_dir, "mros_waveforms_no_resampling/mros-visit1*.zarr/hypnogram"))
    zarr_files_mros_visit1 = list(map(lambda x: str(Path(x).parent), zarr_files_mros_visit1))
    
    zarr_files_mros_visit2 = glob.glob(os.path.join(data_dir, "mros_waveforms_no_resampling/mros-visit2*.zarr/hypnogram"))
    zarr_files_mros_visit2 = list(map(lambda x: str(Path(x).parent), zarr_files_mros_visit2))


    groups_shhs1 = [Path(i).stem.split('-')[-1] for i in zarr_files_shhs1]
    groups_shhs2 = [Path(i).stem.split('-')[-1] for i in zarr_files_shhs2]
    groups_mesa = [Path(i).stem.split('-')[-1] for i in mesa_zarr_files]
    groups_apples = [Path(i).stem.split('-')[-1] for i in apples_zarr_files]
    groups_mros_visit1 = [Path(i).stem.split('-')[-1] for i in zarr_files_mros_visit1]
    groups_mros_visit2 = [Path(i).stem.split('-')[-1] for i in zarr_files_mros_visit2]
    groups_wsc = [Path(i).stem.split('-')[-1] for i in zarr_files_wsc]

    zarr_files = zarr_files_shhs1+zarr_files_shhs2 + zarr_files_mros_visit1+zarr_files_wsc
    external_test_zarrs = mesa_zarr_files+zarr_files_mros_visit2+ apples_zarr_files
    groups = groups_shhs1+groups_shhs2+groups_mros_visit1+groups_wsc

    TRAIN_SIZE = 0.8
    splitter = GroupShuffleSplit(n_splits=1, train_size=TRAIN_SIZE, random_state=random_state)
    train_idxs, test_idxs = next(splitter.split(X=zarr_files, groups=groups))

    train_zarrs = [zarr_files[i] for i in train_idxs]
    test_zarrs = [zarr_files[i] for i in test_idxs]

    groups = [groups[i] for i in train_idxs]
    TRAIN_SIZE = 0.9
    splitter2 = GroupShuffleSplit(n_splits=1, train_size=TRAIN_SIZE, random_state=random_state)
    train_idxs, val_idxs = next(splitter.split(X=train_zarrs, groups=groups))

    val_zarrs = [train_zarrs[i] for i in val_idxs]
    train_zarrs = [train_zarrs[i] for i in train_idxs]


encoder_model = JEPASimpleLightning.load_from_checkpoint(os.path.join(encoder_dir, pretrained_encoder_path), map_location='cpu')

d_model = encoder_model.encoder.d_model
num_heads = encoder_model.encoder.predictor_blocks[0].self_attn.num_heads

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
return_hypnogram_every_sec = config['dataset']['return_hypnogram_every_sec']
overlap = config['dataset']['overlap']
hop_length=win_length - int(overlap*win_length)
max_seq_len_sec = config['dataset']['max_seq_len_sec']
seq_len_sec = sample_stride = max_seq_len_sec
max_seq_len = seq_len_sec*frequency if seq_len_sec is not None else max_seq_len_sec*frequency

n_patches = (max(max_seq_len, win_length)-win_length) // hop_length + 1
if ((max_seq_len-win_length) % hop_length != 0):
    n_patches += 1
n_patches = int(n_patches)


lp_head = dict(c_in=config['lp_head']['c_in'],
                 input_size=d_model, 
                 hidden_size=config['lp_head']['hidden_size'], 
                 n_classes=config['lp_head']['n_classes'],
                 missing_channel_indices=None,
                 module=config['lp_head']['module'], 
                 rnn_dropout=config['lp_head']['rnn_dropout'], 
                 num_rnn_layers=config['lp_head']['num_rnn_layers'], 
                 pool=config['lp_head']['pool'],
                 predict_every_n_patches=config['lp_head']['predict_every_n_patches'],
                 bidirectional=config['lp_head']['bidirectional'], 
                 affine=config['lp_head']['affine'], 
                 pre_norm=config['lp_head']['pre_norm'],
                 mlp_final_head=config['lp_head']['mlp_final_head'],
                 linear_dropout=config['lp_head']['linear_dropout']
                )

lp_model = RNNProbingHead(**lp_head)



name = config['run_config']['name']
filename = f"{name}-sleep_stages" + "{epoch:02d}-CE:{val_ce_loss:.5f}-AUROC:{val_auroc:.5f}-AP:{val_ap:.5f}"
date_str = datetime.now().strftime("%Y-%m-%d")
sub_dir = f"{date_str}-{config['run_config']['model_type']}-{config['run_config']['model_run']}-sleep_stages" if config['run_config']['do_train'] else Path(config['run_config']['test_checkpoint_path']).parent.stem

checkpoint_callback = ModelCheckpoint(dirpath=os.path.join(models_dir, sub_dir), save_top_k=1, monitor="val_loss", mode='min', filename=filename)
checkpoint_callback2 = ModelCheckpoint(dirpath=os.path.join(models_dir, sub_dir), save_top_k=1, monitor="val_auroc", mode='max', filename=filename)

wandb_project = config['run_config']['wandb_project']
wandb_name = f"{date_str}-{config['run_config']['model_type']}-{config['run_config']['model_run']}-{config['run_config']['name']}-sleep_stages"
os.makedirs(os.path.join(models_dir, sub_dir), exist_ok=True)

wandb_logger = WandbLogger(project=f"{wandb_project}", offline=False, name=wandb_name, save_dir=os.path.join(models_dir, sub_dir))
wandb_logger.log_hyperparams({**dict(encoder_model.hparams), **config})

if __name__ == "__main__":
    pl.seed_everything(random_state)

    if Path(os.path.join(models_dir, f'{dataset_filename}-train_samples.csv.gz')).exists():
        train_ds_sample_df = pd.read_csv(os.path.join(models_dir, f'{dataset_filename}-train_samples.csv.gz'))
        train_zarrs = train_ds_sample_df['file'].values.tolist()
    else:
        train_ds_sample_df = None
    train_ds = SelfSupervisedHypnogramTimeDataset(zarr_files=train_zarrs,
                                            channels=channels, 
                                            max_seq_len_sec=max_seq_len_sec, 
                                            sample_seq_len_sec=seq_len_sec, 
                                            sample_stride_sec=sample_stride,
                                            frequency=frequency,
                                            min_seq_len_sec=config['dataset']['min_seq_len_sec'],
                                            start_offset_sec=0,
                                            trim_wake_epochs=config['dataset']['trim_wake_epochs'],
                                            include_partial_samples=config['dataset']['include_partial_samples'], 
                                            sample_df=train_ds_sample_df,
                                            return_hypnogram_every_sec=config['dataset']['return_hypnogram_every_sec'],
                                            hypnogram_padding_mask=config['dataset']['hypnogram_padding_mask'],
                                            hypnogram_frequency=config['dataset']['hypnogram_frequency'],
                                            butterworth_filters=ALL_FREQUENCY_FILTERS,
                                            median_filter_kernel_size=None,
                                            voltage_channels=None,
                                            clip_interpolations=None,
                                            constant_nan_tolerance=config['dataset']['constant_nan_tolerance'],
                                            normalize_signals=config['dataset']['normalize_signals'],
                                            return_hyponogram=True,
                                            hypnogram_required_stages=config['dataset']['hypnogram_required_stages'],
                                            hypnogram_constant_tolerance=config['dataset']['hypnogram_constant_tolerance']
                                            )
    
    if not Path(os.path.join(models_dir, f'{dataset_filename}-train_samples.csv.gz')).exists():
        train_ds_sample_df = train_ds.sample_df
        train_ds_sample_df.to_csv(os.path.join(models_dir, f'{dataset_filename}-train_samples.csv.gz'), index=False, compression='gzip')
    
    if Path(os.path.join(models_dir, f'{dataset_filename}-val_samples.csv.gz')).exists():
        val_ds_sample_df = pd.read_csv(os.path.join(models_dir, f'{dataset_filename}-val_samples.csv.gz'))
        val_zarrs = val_ds_sample_df['file'].values.tolist()
    else:
        val_ds_sample_df = None

    val_ds = SelfSupervisedHypnogramTimeDataset(zarr_files=val_zarrs, 
                                            channels=channels, 
                                            max_seq_len_sec=max_seq_len_sec, 
                                            sample_seq_len_sec=seq_len_sec, 
                                            sample_stride_sec=sample_stride,
                                            frequency=frequency,
                                            min_seq_len_sec=config['dataset']['min_seq_len_sec'],
                                            start_offset_sec=0,
                                            trim_wake_epochs=config['dataset']['trim_wake_epochs'],
                                            include_partial_samples=config['dataset']['include_partial_samples'], 
                                            sample_df=val_ds_sample_df,
                                            return_hypnogram_every_sec=config['dataset']['return_hypnogram_every_sec'],
                                            hypnogram_padding_mask=config['dataset']['hypnogram_padding_mask'],
                                            hypnogram_frequency=config['dataset']['hypnogram_frequency'],
                                            butterworth_filters=ALL_FREQUENCY_FILTERS,
                                            median_filter_kernel_size=None,
                                            voltage_channels=None,
                                            clip_interpolations=None,
                                            normalize_signals=config['dataset']['normalize_signals'],
                                            constant_nan_tolerance=config['dataset']['constant_nan_tolerance'],
                                            return_hyponogram=True,
                                            hypnogram_required_stages=config['dataset']['hypnogram_required_stages'],
                                            hypnogram_constant_tolerance=config['dataset']['hypnogram_constant_tolerance']
                                            )
    
    if not Path(os.path.join(models_dir, f'{dataset_filename}-val_samples.csv.gz')).exists():
        val_ds_sample_df = val_ds.sample_df
        val_ds_sample_df.to_csv(os.path.join(models_dir, f'{dataset_filename}-val_samples.csv.gz'), index=False, compression='gzip')

    if Path(os.path.join(models_dir, f'{dataset_filename}-test_samples.csv.gz')).exists():
        test_ds_sample_df = pd.read_csv(os.path.join(models_dir, f'{dataset_filename}-test_samples.csv.gz'))
        test_zarrs = test_ds_sample_df['file'].values.tolist()
    else:
        test_ds_sample_df = None
    
    test_ds = SelfSupervisedHypnogramTimeDataset(zarr_files=test_zarrs, 
                                            channels=channels, 
                                            max_seq_len_sec=max_seq_len_sec, 
                                            sample_seq_len_sec=seq_len_sec, 
                                            sample_stride_sec=sample_stride,
                                            frequency=frequency,
                                            min_seq_len_sec=config['dataset']['min_seq_len_sec'],
                                            start_offset_sec=0,
                                            trim_wake_epochs=config['dataset']['trim_wake_epochs'],
                                            include_partial_samples=config['dataset']['include_partial_samples'], 
                                            sample_df=test_ds_sample_df,
                                            return_hypnogram_every_sec=config['dataset']['return_hypnogram_every_sec'],
                                            hypnogram_padding_mask=config['dataset']['hypnogram_padding_mask'],
                                            hypnogram_frequency=config['dataset']['hypnogram_frequency'],
                                            butterworth_filters=ALL_FREQUENCY_FILTERS,
                                            median_filter_kernel_size=None,
                                            voltage_channels=None,
                                            clip_interpolations=None,
                                            normalize_signals=config['dataset']['normalize_signals'],
                                            constant_nan_tolerance=config['dataset']['constant_nan_tolerance'],
                                            return_hyponogram=True,
                                            hypnogram_required_stages=config['dataset']['hypnogram_required_stages'],
                                            hypnogram_constant_tolerance=config['dataset']['hypnogram_constant_tolerance']
                                            )
    
    if not Path(os.path.join(models_dir, f'{dataset_filename}-test_samples.csv.gz')).exists():
        test_ds_sample_df = test_ds.sample_df
        test_ds_sample_df.to_csv(os.path.join(models_dir, f'{dataset_filename}-test_samples.csv.gz'), index=False, compression='gzip')

    if Path(os.path.join(models_dir, f'{dataset_filename}-ext-test_samples.csv.gz')).exists():
        ext_test_ds_sample_df = pd.read_csv(os.path.join(models_dir, f'{dataset_filename}-ext-test_samples.csv.gz'))
        ext_test_zarrs = ext_test_ds_sample_df['file'].values.tolist()
    else:
        ext_test_ds_sample_df = None
        ext_test_zarrs = external_test_zarrs

    ext_test_ds = SelfSupervisedHypnogramTimeDataset(zarr_files=ext_test_zarrs, 
                                            channels=channels, 
                                            max_seq_len_sec=max_seq_len_sec, 
                                            sample_seq_len_sec=seq_len_sec, 
                                            sample_stride_sec=sample_stride,
                                            frequency=frequency,
                                            min_seq_len_sec=config['dataset']['min_seq_len_sec'],
                                            start_offset_sec=0,
                                            trim_wake_epochs=config['dataset']['trim_wake_epochs'],
                                            include_partial_samples=config['dataset']['include_partial_samples'], 
                                            sample_df=ext_test_ds_sample_df,
                                            return_hypnogram_every_sec=config['dataset']['return_hypnogram_every_sec'],
                                            hypnogram_padding_mask=config['dataset']['hypnogram_padding_mask'],
                                            hypnogram_frequency=config['dataset']['hypnogram_frequency'],
                                            butterworth_filters=ALL_FREQUENCY_FILTERS,
                                            median_filter_kernel_size=None,
                                            voltage_channels=None,
                                            clip_interpolations=None,
                                            normalize_signals=config['dataset']['normalize_signals'],
                                            constant_nan_tolerance=config['dataset']['constant_nan_tolerance'],
                                            return_hyponogram=True,
                                            hypnogram_required_stages=config['dataset']['hypnogram_required_stages'],
                                            hypnogram_constant_tolerance=config['dataset']['hypnogram_constant_tolerance']
                                            )
    
    if not Path(os.path.join(models_dir, f'{dataset_filename}-ext-test_samples.csv.gz')).exists():
        ext_test_ds_sample_df = ext_test_ds.sample_df
        ext_test_ds_sample_df.to_csv(os.path.join(models_dir, f'{dataset_filename}-ext-test_samples.csv.gz'), index=False, compression='gzip')

    scheduler_kwargs = config['scheduler']
    scheduler_type = config['scheduler']['scheduler_type']
    del scheduler_kwargs['scheduler_type']
    patchmeup_model = PatchTFTSleepStage(learning_rate=config['optimizer']['learning_rate'], 
                                         train_size=len(train_ds), 
                                         batch_size=config['training']['batch_size'],
                                         n_gpus=config['training']['n_gpus'],
                                         linear_probing_head=lp_model,
                                         preloaded_model=encoder_model,
                                         metrics=metrics,
                                         fine_tune=False,
                                         loss_fxn=loss_fxn,
                                         class_weights=None,
                                         gamma=2.,
                                         label_smoothing=0.,
                                         y_padding_mask=config['dataset']['hypnogram_padding_mask'],
                                         epochs=config['training']['epochs'], 
                                         weight_decay=config['optimizer']['weight_decay'],
                                         use_weight_decay_scheduler=config['optimizer']['use_weight_decay_scheduler'],
                                         final_weight_decay=config['optimizer']['final_weight_decay'],
                                         optimizer_type=config['optimizer']['optimizer_type'],
                                         scheduler_type=scheduler_type,
                                         scheduler_kwargs=scheduler_kwargs
                                         )

    train_loader = DataLoader(train_ds, batch_size=config['training']['batch_size'], shuffle=True, num_workers=config['training']['num_workers'], drop_last=True, persistent_workers=True, pin_memory=False, collate_fn=nested_tensor_sequence_collate)
    val_loader = DataLoader(val_ds, batch_size=config['training']['batch_size'], shuffle=False, num_workers=config['training']['num_workers'], persistent_workers=True, pin_memory=False, collate_fn=nested_tensor_sequence_collate)
    test_loader = DataLoader(test_ds, batch_size=config['training']['batch_size'], shuffle=False, num_workers=config['training']['num_workers'], persistent_workers=True, pin_memory=False, collate_fn=nested_tensor_sequence_collate)
    ext_test_loader = DataLoader(ext_test_ds, batch_size=config['training']['batch_size'], shuffle=False, num_workers=config['training']['num_workers'], persistent_workers=True, pin_memory=False, collate_fn=nested_tensor_sequence_collate)

    trainer = pl.Trainer(precision=config['run_config']['precision'],
                     enable_checkpointing=True,
                     enable_progress_bar=True,
                     enable_model_summary=True,
                     logger=wandb_logger,
                     strategy="ddp",
                     gradient_clip_val=config['training']['gradient_clip_val'],
                     gradient_clip_algorithm='norm' if config['training']['use_gradient_clipping'] else None,
                     log_every_n_steps=50,
                     num_sanity_val_steps=2,
                     detect_anomaly=False,
                     profiler=None,
                     accelerator="gpu", 
                     accumulate_grad_batches=config['training']['accumulate_grad_batches'],
                     devices=config['training']['n_gpus'], 
                     default_root_dir=models_dir, 
                     max_epochs=config['training']['epochs'], 
                     fast_dev_run=False,
                     callbacks=[checkpoint_callback, checkpoint_callback2])
    


    if config['run_config']['do_train']:
        trainer.fit(model=patchmeup_model, train_dataloaders=train_loader, val_dataloaders=val_loader)
        best_model_path = checkpoint_callback2.best_model_path 
    else:
        assert os.path.exists(config['run_config']['test_checkpoint_path']), "Provide path to trained model checkpoint for testing"
        best_model_path = config['run_config']['test_checkpoint_path']

    patchmeup_model = PatchTFTSleepStage.load_from_checkpoint(best_model_path, 
                                                             preloaded_model=encoder_model,
                                                             linear_probing_head=lp_model,
                                                             map_location='cpu')
    val_preds,val_targets = get_predictions_nested(data_loader=val_loader, model=patchmeup_model, dataloader_name='val')
    test_preds,test_targets = get_predictions_nested(data_loader=test_loader, model=patchmeup_model, dataloader_name='test')
    ext_test_preds,ext_test_targets = get_predictions_nested(data_loader=ext_test_loader, model=patchmeup_model, dataloader_name='ext_test')
    expected_size = int(max_seq_len_sec / return_hypnogram_every_sec)

    val_targets_cat = torch.concat([i.to_padded_tensor(config['dataset']['hypnogram_padding_mask'], output_size=(1,expected_size)) for i in val_targets]).cpu()
    test_targets_cat = torch.concat([i.to_padded_tensor(config['dataset']['hypnogram_padding_mask'], output_size=(1,expected_size)) for i in test_targets]).cpu()
    ext_test_targets_cat = torch.concat([i.to_padded_tensor(config['dataset']['hypnogram_padding_mask'], output_size=(1,expected_size)) for i in ext_test_targets]).cpu()
    
    val_preds_cat = torch.cat([F.pad(i, (0, expected_size - i.shape[-1]), "constant", 0) for i in val_preds]).cpu()
    test_preds_cat = torch.cat([F.pad(i, (0, expected_size - i.shape[-1]), "constant", 0) for i in test_preds]).cpu()
    ext_test_preds_cat = torch.cat([F.pad(i, (0, expected_size - i.shape[-1]), "constant", 0) for i in ext_test_preds]).cpu()


    

    tensor_dict = {'val_targets': val_targets_cat, 'val_preds': val_preds_cat,
                   'test_targets': test_targets_cat, 'test_preds': test_preds_cat,
                   'ext_test_targets':ext_test_targets_cat, 'ext_test_preds': ext_test_preds_cat
                   }
    if config['run_config']['do_train']:
        torch.save(tensor_dict, os.path.join(models_dir, sub_dir, f'sleep_stages-{config["run_config"]["name"]}-predictions.pt'))
    else:
        dir_ = os.path.dirname(config['run_config']['test_checkpoint_path'])
        torch.save(tensor_dict, os.path.join(dir_, f'sleep_stages-{config["run_config"]["name"]}-predictions.pt'))