import os
import pandas as pd
from pathlib import Path

from src.dataset.base_dataset import BaseAudioFakeDataset

class ASVspoofDataset(BaseAudioFakeDataset):
    def __init__(self, root_dir, subset='train'):
        super().__init__(subset)
        
        self.root_dir = Path(root_dir)

        self.read_samples()
        
    def read_samples(self):
        if self.subset == 'train':
            data_dir = self.root_dir / 'ASVspoof2019_LA_train'
            protocol_file = self.root_dir / 'ASVspoof2019_LA_cm_protocols' / 'ASVspoof2019.LA.cm.train.trn.txt'
        elif self.subset == 'dev':
            data_dir = self.root_dir / 'ASVspoof2019_LA_dev'
            protocol_file = self.root_dir / 'ASVspoof2019_LA_cm_protocols' / 'ASVspoof2019.LA.cm.dev.trl.txt'
        elif self.subset == 'eval':
            data_dir = self.root_dir / 'ASVspoof2019_LA_eval'
            protocol_file = self.root_dir / 'ASVspoof2019_LA_cm_protocols' / 'ASVspoof2019.LA.cm.eval.trl.txt'
        else:
            raise ValueError(f"Invalid subset: {self.subset}. Choose from 'train', 'dev', or 'eval'.")
        
        # Load protocol file
        protocol_df = pd.read_csv(protocol_file, sep=' ', header=None,
                                  names=['speaker_id', 'file_name', 'system_id', '_', 'label'])
        
        # Fill samples dataframe
        self.samples = pd.DataFrame({
            'path': protocol_df['file_name'].apply(lambda x: str(data_dir / "flac" / f"{x}.flac")),
            'label': protocol_df['label'],
            'speaker_id': protocol_df['speaker_id'],
        })
        
        # Verify all files exist
        missing_files = self.samples[~self.samples['path'].apply(os.path.exists)]
        if not missing_files.empty:
            raise FileNotFoundError(f"Missing {len(missing_files)} files. Ex: {missing_files['path'].head().tolist()}")


if __name__ == "__main__":

    for split in ["train", "dev", "eval"]:
        print(split)

        dataset = ASVspoofDataset('/home/jnb5885/deepfake/ASVspoof19_LA', subset=split)

        print(len(dataset))
        print(dataset.samples['label'].value_counts())