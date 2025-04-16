import argparse
import logging
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import yaml
import torch

from src.dataset import datasets
from src.dataset.base_dataset import BaseAudioFakeDataset
from src.models import models
from src.trainer import Trainer
from src.util import set_seed
import os
def seed_everything(seed: int):
    import random, os
    import numpy as np
    import torch

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

def setup_logger(model_id: str, save_file_id: str) -> None:
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)

    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File handler
    log_folder = Path("output/log")
    log_filename = log_folder / f"{model_id}_{save_file_id}.txt"
    try:
        log_folder.mkdir(parents=True, exist_ok=True)
        logger.info(f"Log folder created/verified at: {log_filename}")
    except Exception as e:
        logger.error(f"Failed to create log folder: {e}")
        return

    fh = logging.FileHandler(log_filename)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

def save_model(
    save_dir: Path, 
    model: torch.nn.Module, 
    config: Dict,
    metrics: Dict,
) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)

    # Save model checkpoint
    ckpt_path = save_dir / "ckpt.pth"
    torch.save(model.state_dict(), ckpt_path)

    # Save config
    config["checkpoint"] = {"path": str(ckpt_path)}
    config_path = save_dir / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)

    # Save training metrics
    metrics_path = save_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)

def load_data_split(
    dataset_name: str,
    embedding_path: str
) -> Tuple[BaseAudioFakeDataset, BaseAudioFakeDataset]:
    with open("./paths.conf", "r") as f:
        data_paths = json.load(f)
    logging.info("Loaded dataset paths conf")

    data_train = datasets.get_dataset(dataset_name, embedding_path, data_paths, subset="train")
    data_test = datasets.get_dataset(dataset_name, embedding_path, data_paths, subset="val")
    logging.info(f"Loaded '{dataset_name}' data split")

    return data_test, data_train

def train(
    device: str,
    config: Dict,
    data_split: Tuple[BaseAudioFakeDataset, BaseAudioFakeDataset],
    model_save_dir: Path
) -> None:
    model_config = config["model"]
    model_class, model_parameters = model_config["class"], model_config["parameters"]
    train_config = config["training"]

    data_test, data_train = data_split

    logging.info(f"Initializing model '{model_class}' on device {device}")
    current_model = models.get_model(
        model_name=model_class,
        config=model_parameters,
        device=device,
    )

    # Load pre-trained weights if specified
    init_weights_path = train_config.get("init_weights", "")
    if init_weights_path:
        logging.info(f"Loading pre-trained weights from {init_weights_path}")
        current_model.load_state_dict(torch.load(init_weights_path))
        logging.info(f"Finetuning '{model_class}' model on {len(data_train)} audio files.")
    else:
        logging.info(f"Training '{model_class}' model from scratch on {len(data_train)} audio files.")
    logging.info(f"Testing on {len(data_test)} audio files")

    # Freeze encoder parameters if specified
    if model_parameters.get("freeze_encoder"):
        for param in current_model.whisper_model.parameters():
            param.requires_grad = False

    current_model = current_model.to(device)

    # Training
    use_scheduler = "rawnet3" in model_class.lower()
    use_context = "context" in model_class.lower()

    trainer = Trainer(
        device=device,
        batch_size=train_config['batch_size'],
        epochs=train_config['max_epochs'],
        optimizer_kwargs=train_config['optimzer'],
        use_scheduler=use_scheduler,
        use_context=use_context,
    )
    current_model, metrics = trainer.train(
        model=current_model,
        train_dataset=data_train,
        test_dataset=data_test,
    )

    # Save trained model and config
    logging.info(f"Saving model to {model_save_dir}")
    save_model(model_save_dir, current_model, config, metrics)

    # Save predictions of best model on test set
    # logging.info("Evaluating best model on test set")
    # _, test_loader = trainer._init_loaders(data_train, data_test)
    # _, all_preds, all_labels = trainer._evaluate(current_model, test_loader, return_preds=True)
    # saved_pred = pd.DataFrame({
    #     'pred': all_preds.squeeze(1),
    #     'pred_label': (all_preds >= 0.5).astype(int).squeeze(1),
    #     'true': all_labels.astype(int).squeeze(1),
    # })
    # saved_pred_path = f"{model_save_dir}/test_pred.csv"
    # logging.info(f"Saving test predictions to {saved_pred_path}")
    # saved_pred.to_csv(saved_pred_path, index=False)

def main(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    print(args)
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    save_file_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_id = config['model'].get("id", "")
    epoch = config["training"]["max_epochs"]
    Traindata_name = config["data"]["name"]
    if args.embedding_path is None:
        model_save_dir = Path(f"models/{model_id}_seed_{args.seed}_epoch_{epoch}_train_{Traindata_name}")
    else:
        context_name = args.embedding_path.split("/")[-1].split("_")[-1]
        model_save_dir = Path(f"models/{model_id}_context_{context_name}_seed_{args.seed}_epoch_{epoch}_train_{Traindata_name}")

    if os.path.exists(model_save_dir):
        raise Exception("Have Trained")


    setup_logger(model_id, save_file_id)
    logging.info(f"Loaded config: {args.config}")
    logging.info(f"Saving model to dir: {model_save_dir}")
    
    dataset_name = config['data']['name']
    data_split = load_data_split(dataset_name, args.embedding_path)

    train(
        device=args.device,
        config=config,
        data_split=data_split,
        model_save_dir=model_save_dir
    )

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        help="Model config file path",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--device",
        "-d",
        help="Device name (default: cuda).",
        type=str,
        default='cuda',
    )
    parser.add_argument(
        "--embedding_path",
        "-emb",
        help=f"path to jdd new embedding with fixed dimensions",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--seed",
        "-s",
        help="random seed",
        type=int,
        default='1',
    )
    return parser.parse_args()

if __name__ == "__main__":
    main(parse_args())

    # --config configs/whisper_specrnet_inthewild_new_test_context.yaml --embedding_path /tank/local/cgo5577/dataset_jdd/DATASETS_ITW_FOR_CHONGYANG/CT
    # --config configs/whisper_specrnet_inthewild_new_test.yaml