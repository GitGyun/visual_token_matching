import os
import pytorch_lightning as pl
import torch

from train.trainer import LightningTrainWrapper
from train.train_utils import configure_experiment, get_ckpt_path


if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    IS_RANK_ZERO = int(os.environ.get('LOCAL_RANK', 0)) == 0
    
    from args import config # parse arguments

    # environmental settings
    logger, save_dir, callbacks, profiler, precision, strategy = configure_experiment(config)

    # create pytorch lightning trainer.
    trainer = pl.Trainer(
        logger=logger,
        default_root_dir=save_dir,
        accelerator='gpu',
        max_epochs=((config.n_steps // config.val_iter) if not config.no_eval else 1),
        log_every_n_steps=-1,
        num_sanity_val_steps=(2 if config.sanity_check else 0),
        callbacks=callbacks,
        benchmark=True,
        devices=-1,
        strategy=strategy,
        precision=precision,
        profiler=profiler,
    )

    # load checkpoint to continue.
    if config.continue_mode:
        ckpt_path = get_ckpt_path(config)
    else:
        ckpt_path = None
    model = LightningTrainWrapper(config, verbose=IS_RANK_ZERO)
    
    # start training.
    trainer.fit(model, ckpt_path=ckpt_path)
