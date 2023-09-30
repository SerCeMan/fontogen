import lightning.pytorch as pl
import torch
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger

from config import fontogen_config
from model.model import FontogenModule, FontogenDataModule
from sampler import SamplingCallback


# Install the magic combination of nightly deps.
#
# pip install -U --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/Triton-Nightly/pypi/simple/ triton-nightly==2.1.0.dev20230801015042 --no-deps
#

def main():
    config = fontogen_config()

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.has_mps else "cpu"
    precision = "bf16-mixed" if device == "cuda" else 32
    use_wandb = True
    checkpoint_path = None
    dataset_path = 'data/combined_glyphs_4_3.ds'

    torch.set_float32_matmul_precision("medium")

    model = FontogenModule(config) if checkpoint_path is None else FontogenModule.load_from_checkpoint(
        checkpoint_path,
        config=config,
    )

    data_module = FontogenDataModule(config, dataset_path)

    checkpoint_callback = ModelCheckpoint(
        save_last=True,
        save_top_k=10,
        monitor='val_loss',
    )

    lr_monitor = LearningRateMonitor(logging_interval='step')
    num_epochs = 200

    trainer = pl.Trainer(
        max_epochs=num_epochs,
        accelerator=device,
        accumulate_grad_batches=16,
        gradient_clip_val=0.5,
        precision=precision,
        # enable_checkpointing=False,
        callbacks=[
            checkpoint_callback,
            lr_monitor,
            SamplingCallback(config, 10, 'training/samples'),
        ],
        logger=WandbLogger(project='fontogen3', dir='training') if use_wandb else None,
    )
    trainer.fit(
        model,
        data_module,
        # ckpt_path=checkpoint_path,
    )


if __name__ == "__main__":
    main()
