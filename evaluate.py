import os
import argparse
import yaml
import json
from typing import Dict, Tuple
from datetime import datetime

import pandas as pd

import torch

from src.dataset import datasets
from src.dataset.base_dataset import BaseAudioFakeDataset
from src.models import models
from src.trainer import Trainer
from src.util import set_seed
import time
BATCH_SIZE = 16

def load_model(model_folder: str, device: str) -> Tuple[torch.nn.Module, Dict]:
    set_seed(42)

    print(f"Loading from {model_folder} onto device {device}")

    with open(f"{model_folder}/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    model_config = config["model"]
    model_class, model_parameters = model_config["class"], model_config["parameters"]

    print(f"Initializing model '{model_class}' on device {device}")
    current_model = models.get_model(
        model_name=model_class,
        config=model_parameters,
        device=device,
    )
    
    print(f"Loading pre-trained weights from '{config['checkpoint']['path']}'")
    current_model.load_state_dict(torch.load(config["checkpoint"]["path"]))
    current_model.to(device)

    return current_model, config

def load_data_split(
    dataset_name: str,
    embedding_path: str
) -> Tuple[BaseAudioFakeDataset, BaseAudioFakeDataset]:
    with open("./paths.conf", "r") as f:
        data_paths = json.load(f)
    print("Loaded dataset paths conf")
    data_train = datasets.get_dataset(dataset_name, embedding_path, data_paths, subset="train")
    data_test = datasets.get_dataset(dataset_name, embedding_path, data_paths, subset="test")
    data_val = datasets.get_dataset(dataset_name, embedding_path, data_paths, subset="val")
    print(f"Loaded '{dataset_name}' data split")
    return data_test, data_train

# def load_data_split(
#     dataset_name: str
# ) -> Tuple[BaseAudioFakeDataset, BaseAudioFakeDataset]:
#     with open("./paths.conf", "r") as f:
#         data_paths = json.load(f)
#     print("Loaded dataset paths conf")
#
#     data_train = datasets.get_dataset(dataset_name, data_paths, subset="train")
#     data_test = datasets.get_dataset(dataset_name, data_paths, subset="test")
#     print(f"Loaded '{dataset_name}' data split")
#
#     return data_test, data_train

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained models.")
    parser.add_argument('--device', type=str, default='cuda', help='Device to use for evaluation (e.g., "cuda" or "cpu").')
    parser.add_argument('--model-path', type=str, help='Path to model folder (or folder containing model subfolders).')
    parser.add_argument('--group', action='store_true', help='If set, evaluate all subfolders within model folder.')
    parser.add_argument('--dataset', type=str, default='jdd', help='Name of deepfake dataset.')
    parser.add_argument('--embedding_path', type=str, default=None, help='Name of deepfake dataset.')
    # parser.add_argument('--save-pred', action='store_true', help='If set, saves model predictions on test set.')


    args = parser.parse_args()

    model_prefix = args.model_path.split("/")[-1]
    save_name = f"./results_base/{args.dataset}_{model_prefix}"

    # Save with unique name
    save_file_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    start_time = time.time()
    # Loading data
    data_test, data_train = load_data_split(args.dataset, args.embedding_path)

    # Selecting models for evaluation
    all_model_paths = []
    if args.group:
        for subfolder in os.listdir(args.model_path):
            full_path = os.path.join(args.model_path, subfolder)
            if os.path.isdir(full_path):
                all_model_paths.append(full_path)
    else:
        all_model_paths = [args.model_path]
    print(f"Evaluating {len(all_model_paths)} models")

    # Evaluation
    all_metrics = []

    for i, path in enumerate(all_model_paths):
        print(f"Loading model {i+1}/{len(all_model_paths)}")    
        current_model, current_config = load_model(path, args.device)

        model_class = current_config["model"]["class"]
        use_context = "context" in model_class.lower()
        print(f"use_context set: {use_context}")
        harness = Trainer(
            device=args.device,
            batch_size=BATCH_SIZE,
            use_context=use_context
        )

        # Use trainer class as harness for evaluation
        _, test_loader = harness._init_loaders(data_train, data_test)
        metrics = harness._evaluate(current_model, test_loader)

        metrics['model'] = current_config['model']['id']
        metrics['ckpt_folder'] = path
        metrics['eval_dataset'] = args.dataset
        metrics['train_dataset'] = current_config['data']['name']
        all_metrics.append(metrics)

        print(f"Summary: Acc: {metrics['accuracy']:.4f}, EER: {metrics['eer']:.4f}, AUC: {metrics['auc']:.4f}")

        end_time = time.time()

        training_time = end_time - start_time

        print("Test time:", training_time, "seconds")

        test_time = [training_time, training_time / (BATCH_SIZE*len(test_loader))]

        time_save_dir = f"./results_base/infer_time_{args.dataset}_{model_prefix}.json"
        with open(time_save_dir, "w") as f:
            json.dump(test_time, f, indent=4)



    results = pd.DataFrame(all_metrics)
    # improving results formatting
    cols = ['model'] + [col for col in results.columns if col != 'model']
    results = results[cols]
    results = results.sort_values('model')
    # save to csv
    results.to_csv(f"./{save_name}.csv", index=False)

