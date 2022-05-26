import argparse
import re
from datetime import datetime

import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

import wandb
from src.datasets import get_loaders
from src.models import load_model
from src.utils.evaluation import get_classification_scores
from src.utils.run import make_deterministic


def main(conf: DictConfig, config_name: str):
    run_name = f"{config_name}_{re.sub('[^0-9]', '', str(datetime.now()))}"
    wandb.init(name=run_name, project="bsc", config=conf)

    model = load_model(conf)

    train_loader, val_loader, test_loader = get_loaders(conf)

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath="./model/",
        filename="model-{epoch:02d}-{val_loss:.2f}",
        save_top_k=2,
        mode="min",
    )

    wandb_logger = WandbLogger()

    trainer = pl.Trainer(
        callbacks=[checkpoint_callback], max_epochs=conf.train_epochs, logger=wandb_logger
    )

    trainer.fit(model, train_loader, val_loader)

    model = load_model(conf=conf, model_path=checkpoint_callback.best_model_path)
    # Evaluating trained model
    predictions = model.predict(test_loader)

    wandb.log(get_classification_scores(predictions))
    wandb.finish()


parser = argparse.ArgumentParser()
parser.add_argument("--data-dir", type=str, default="./data/", help="A base dataset directory")
parser.add_argument(
    "--split-num",
    type=int,
    default=1,
    help="Dataset split number used for experimentation",
)
parser.add_argument("--experiment", type=str, help="Experiment name")
parser.add_argument("--model-name", type=str, help="Experiment name")
parser.add_argument("--device", type=str, help="Device to run model on")

if __name__ == "__main__":
    parser_args = parser.parse_args()
    config_filename = (
        f"./config/experiments/{parser_args.experiment}/{parser_args.model_name}.yaml"
    )
    config_name = (
        re.search("config/(.*).yaml", config_filename).group(1) + f"_split_{parser_args.split_num}"
    )

    config = OmegaConf.load(config_filename)
    config["data_dir"] = parser_args.data_dir
    config["load_from_dir"] = config["load_from_dir"] + f"/split_{parser_args.split_num}"
    if parser_args.device:
        config["device"] = parser_args.device

    make_deterministic(seed=config.seed)
    main(config, config_name)
