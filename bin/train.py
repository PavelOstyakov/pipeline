from pipeline.logger import setup_logger
from pipeline.utils import load_config

from torch.utils.data import DataLoader

import argparse
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path")
    args = parser.parse_args()

    config = load_config(args.config_path)

    train_data_loader = DataLoader(
        config.train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=config.num_workers)

    val_data_loader = DataLoader(
        config.val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers)

    model = config.model

    model_save_path = config.model_save_path
    os.makedirs(model_save_path, exist_ok=True)

    logger_path = os.path.join(model_save_path, "log.txt")

    setup_logger(out_file=logger_path)

    trainer = config.trainer_cls(
        model=model,
        train_data_loader=train_data_loader,
        val_data_loader=val_data_loader,
        epoch_count=config.epoch_count,
        optimizer=config.optimizer,
        scheduler=config.scheduler,
        loss_calculator=config.loss_calculator,
        metric_calculator=config.metric_calculator,
        print_frequency=config.print_frequency
    )

    trainer.run()


if __name__ == "__main__":
    main()
