import lightning.pytorch as pl
import torch
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
    RichProgressBar,
    RichModelSummary,
)
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.strategies import DDPStrategy

from src.WADEPre import WADEPre

torch.set_float32_matmul_precision("medium")

MODEL_NAME = "WADEPre"


def main():
    # seed
    pl.seed_everything(42, workers=True)

    # callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath="checkpoints",
            filename="{epoch}-{val/loss:.2f}",
            monitor="val/csi_mean",
            mode="max",
            save_last=True,
            save_top_k=3,
            save_on_train_epoch_end=False
        ),
        LearningRateMonitor(logging_interval="step", log_weight_decay=False),
        RichProgressBar(),
        RichModelSummary(max_depth=3),
        # EarlyStopping(monitor="val/loss", patience=5, mode="min"),
    ]

    # logger
    csv_logger = CSVLogger(
        save_dir="logs", name=MODEL_NAME, flush_logs_every_n_steps=10
    )

    # init model
    m = WADEPre(
        timesteps=6,
        spatial_size=128,
        loss_a_stop_step=3000,
        lr=1.5e-4,
        wavelet_level=3,
        detail_layer_channels=[64, 128, 256],
        detail_num_blocks=4,
        loss_a_weight=0.1,
        loss_a_constant_weight=0.01,
        loss_d_weight=0.05,
        loss_recon_mean_weight=0.005,
        detail_idr_dim=64,
        detail_feature_channel=128,
        refine_hidden_dim=6 * 96,
        approx_hidden_size=512,
        approx_cells=3,
        dropout_rate = 0.1
    )

    # trainer
    trainer = pl.Trainer(
        max_epochs=200,
        accelerator="gpu",
        devices=4,
        precision="32",
        log_every_n_steps=30,
        enable_model_summary=True,
        callbacks=callbacks,
        logger=[csv_logger],
        gradient_clip_val=1.0,
        strategy=DDPStrategy(find_unused_parameters=False),
    )

    trainer.fit(model=m, datamodule=dataset)



if __name__ == "__main__":
    main()
