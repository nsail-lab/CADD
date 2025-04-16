from typing import Dict
import numpy as np

from src.dataset import (
    base_dataset,
    asvspoof_dataset,
    in_the_wild_dataset,
    jdd_dataset,
)
from src.util import IN_THE_WILD_TEST_IDS_PATH

def get_dataset(dataset_name: str, embedding_path: str, paths: Dict,  subset: str) -> base_dataset.BaseAudioFakeDataset:
    if dataset_name == "asvspoof19":
        # ASVspoof19 uses split names ['train', 'dev', 'eval']
        subset = 'dev' if subset == 'test' else subset
        return asvspoof_dataset.ASVspoofDataset(
            root_dir=paths["asvspoof_path"],
            subset=subset,
        )
    elif dataset_name == "in_the_wild":
        # Load test ids to create split
        # assert subset in ["train", "test"], "invalid subset name for in_the_wild"
        # test_ids = np.load(IN_THE_WILD_TEST_IDS_PATH)
        # IN_THE_WILD_TOTAL_SAMPLES = 31779
        # train_ids = np.setdiff1d(np.arange(IN_THE_WILD_TOTAL_SAMPLES), test_ids)
        if "meta_file" in paths.keys():
            return in_the_wild_dataset.InTheWildDataset(
                root_dir=paths["in_the_wild_context_path"],
                subset=subset,
                meta_name=paths["meta_file"],
            )
        else:
            return in_the_wild_dataset.InTheWildDataset(
                root_dir=paths["in_the_wild_context_path"],
                subset=subset,
                meta_name="meta_processed.csv",
            )
    elif dataset_name == "in_the_wild_context":
        # Load test ids to create split
        # assert subset in ["train", "test"], "invalid subset name for in_the_wild"
        # test_ids = np.load(IN_THE_WILD_TEST_IDS_PATH)
        # IN_THE_WILD_TOTAL_SAMPLES = 31779
        # train_ids = np.setdiff1d(np.arange(IN_THE_WILD_TOTAL_SAMPLES), test_ids)
        if "meta_file" in paths.keys():
            return in_the_wild_dataset.InTheWildContextDataset(
                root_dir=paths["in_the_wild_context_path"],
                subset=subset,
                embedding_path=embedding_path,
                meta_name=paths["meta_file"],
            )
        else:
            return in_the_wild_dataset.InTheWildContextDataset(
                root_dir=paths["in_the_wild_context_path"],
                subset=subset,
                embedding_path=embedding_path,
                meta_name='meta_processed.csv',
            )
    elif dataset_name == "jdd":
        if "meta_file" in paths.keys():
            return jdd_dataset.jddDataset(
                root_dir=paths["jdd_path"],
                subset=subset,
                meta_name=paths["meta_file"],
            )
        else:
            return jdd_dataset.jddDataset(
                root_dir=paths["jdd_path"],
                subset=subset,
                meta_name='meta_audio.csv',
            )
    elif dataset_name == "jdd_context":
        if "meta_file" in paths.keys():
            return jdd_dataset.jddContextDataset(
                root_dir=paths["jdd_path"],
                embedding_path=embedding_path,
                subset=subset,
                meta_name=paths["meta_file"],
            )
        else:
            return jdd_dataset.jddContextDataset(
                root_dir=paths["jdd_path"],
                embedding_path=embedding_path,
                subset=subset,
                meta_name='meta_audio.csv',
            )
    elif dataset_name == "jdd_synthetic":
        return jdd_dataset.jddDataset(
            root_dir=paths['jdd_synthetic_path'],
            subset=subset,
        )
    elif dataset_name == "jdd_synthetic_context":
        return jdd_dataset.jddContextDataset(
            root_dir=paths['jdd_synthetic_path'],
            embedding_path=embedding_path,
            subset=subset,
        )
    elif dataset_name == "jdd_boost":
        return jdd_dataset.jddDataset(
            root_dir=paths["jdd_boost_path"],
            subset=subset,
            meta_name='meta.csv',
        )
    elif dataset_name == "jdd_boost_context":
        return jdd_dataset.jddContextDataset(
            root_dir=paths["jdd_boost_path"],
            embedding_path=embedding_path,
            subset=subset,
            meta_name='meta.csv',
        )
    else:
        raise ValueError(f"Dataset '{dataset_name}' not supported")


def get_dataset_noise(dataset_name: str, embedding_path: str, paths: str, data_dir: str,  subset: str) -> base_dataset.BaseAudioFakeDataset:
    if dataset_name == "asvspoof19":
        # ASVspoof19 uses split names ['train', 'dev', 'eval']
        subset = 'dev' if subset == 'test' else subset
        return asvspoof_dataset.ASVspoofDataset(
            root_dir=paths["asvspoof_path"],
            subset=subset,
        )
    elif dataset_name == "in_the_wild":
        return in_the_wild_dataset.InTheWildDataset(
            root_dir=paths,
            subset=subset,
            meta_name="meta_processed.csv",
            data_dir=data_dir,
        )
    elif dataset_name == "in_the_wild_context":
        return in_the_wild_dataset.InTheWildContextDataset(
            root_dir=paths,
            subset=subset,
            embedding_path=embedding_path,
            meta_name='meta_processed.csv',
            data_dir=data_dir,
        )
    elif dataset_name == "jdd":
        return jdd_dataset.jddDataset(
            root_dir=paths,
            subset=subset,
            meta_name='meta_audio.csv',
            data_dir=data_dir,
        )
    elif dataset_name == "jdd_context":
        return jdd_dataset.jddContextDataset(
            root_dir=paths,
            embedding_path=embedding_path,
            subset=subset,
            meta_name='meta_audio.csv',
            data_dir=data_dir,
        )
    elif dataset_name == "jdd_synthetic":
        return jdd_dataset.jddDataset(
            root_dir=paths,
            subset=subset,
            meta_name='meta.csv',
            data_dir=data_dir,
        )
    elif dataset_name == "jdd_synthetic_context":
        return jdd_dataset.jddContextDataset(
            root_dir=paths,
            embedding_path=embedding_path,
            subset=subset,
            meta_name='meta.csv',
            data_dir=data_dir,
        )
    # elif dataset_name == "jdd_boost":
    #     return jdd_dataset.jddDataset(
    #         root_dir=paths["jdd_boost_path"],
    #         subset=subset,
    #         meta_name='meta.csv',
    #     )
    # elif dataset_name == "jdd_boost_context":
    #     return jdd_dataset.jddContextDataset(
    #         root_dir=paths["jdd_boost_path"],
    #         embedding_path=embedding_path,
    #         subset=subset,
    #         meta_name='meta.csv',
    #     )
    else:
        raise ValueError(f"Dataset '{dataset_name}' not supported")