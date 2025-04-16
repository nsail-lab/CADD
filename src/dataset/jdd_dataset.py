from pathlib import Path
import pandas as pd
import numpy as np
import json
import torch
from torch.utils.data.dataset import T_co

from src.dataset.base_dataset import BaseAudioFakeDataset
import torchaudio
import pickle
SAMPLING_RATE = 16_000
APPLY_NORMALIZATION = True
APPLY_TRIMMING = True
APPLY_PADDING = True
FRAMES_NUMBER = 480_000  # <- originally 64_600

SOX_SILENCE = [
    # trim all silence that is longer than 0.2s and louder than 1% volume (relative to the file)
    # from beginning and middle/end
    ["silence", "1", "0.2", "1%", "-1", "0.2", "1%"],
]

class jddDataset(BaseAudioFakeDataset):
    def __init__(
        self,
        root_dir,
        subset=None,
        sub_ids=None,
        meta_name='meta.csv',
        data_dir='data',
    ):
        super().__init__(subset=subset)
        self.root_dir = root_dir
        self.meta_name = meta_name
        self.data_dir = data_dir

        self.read_samples()
        # sub_ids allow for testing on specific subset of data
        if sub_ids is not None:
            self.samples = self.samples.iloc[sub_ids]

    def read_samples(self):
        path = Path(self.root_dir)
        meta_path = path / self.meta_name

        self.samples = pd.read_csv(meta_path)
        self.samples["path"] = self.samples["file_name"].apply(lambda n: str(path / self.data_dir / f"{n}.wav"))
        # self.samples["path"] = self.samples["file_name"].apply(lambda n: str(path / "data" / f"{n}.wav"))
        self.samples["label"] = self.samples["label"].map({"real": "bonafide", "fake": "spoof"})

        # Keep only the samples for the specified subset
        if self.subset is not None:
            self.samples = self.samples[self.samples["split"] == self.subset]


class jddContextDataset(BaseAudioFakeDataset):
    def __init__(
            self,
            root_dir,
            subset=None,
            embedding_path="/tank/local/cgo5577/deepfake-whisper/processed_embedding",
            meta_name='meta.csv',
            data_dir='data',
    ):
        super().__init__(subset=subset)
        self.root_dir = root_dir
        self.meta_name = meta_name
        self.embedding_path = embedding_path
        self.data_dir = data_dir

        self.read_samples()
        # sub_ids allow for testing on specific subset of data


    def read_samples(self):
        path = Path(self.root_dir)
        meta_path = path / self.meta_name

        samples = pd.read_csv(meta_path)
        samples = samples.to_dict(orient='records')
        samples_data = []
        for i in range(len(samples)):
            # if samples[i]["file_name"] == "Emma_StonePodcastClip" or samples[i]["file_name"] == "RonDeSantisInsta":
            #     # Emma_StonePodcastClip RonDeSantisInsta
            #     continue
            if samples[i]["split"] == self.subset:
                samples_data.append(samples[i])

        self.samples = []
        embedding_path = Path(self.embedding_path)

        for i in range(len(samples_data)):
            sample = {}
            sample["path_audio"] = str(path / self.data_dir / (samples_data[i]["file_name"] + ".wav"))
            sample["path_embedding"] = str(embedding_path /  (samples_data[i]["file_name"] + ".pkl"))
            sample["label"] = samples_data[i]["label"]
            sample["speaker"] = samples_data[i]["subjects"]

            self.samples.append(sample)

    def __getitem__(self, index) -> T_co:
        label = self.samples[index]["label"]

        waveform, sample_rate = torchaudio.load(self.samples[index]["path_audio"], normalize=APPLY_NORMALIZATION)
        real_sec_length = len(waveform[0]) / sample_rate

        waveform, sample_rate = apply_preprocessing(waveform, sample_rate)

        with open(self.samples[index]["path_embedding"], 'rb') as f:
            context = pickle.load(f)

        # (a-avg)/std
        context_e = torch.from_numpy(context)

        return_data = [waveform, sample_rate, context_e]


        label = 1 if label == "real" else 0

        return (waveform, context_e), label

def apply_preprocessing(
    waveform,
    sample_rate,
):
    if sample_rate != SAMPLING_RATE and SAMPLING_RATE != -1:
        waveform, sample_rate = resample_wave(waveform, sample_rate, SAMPLING_RATE)

    # Stereo to mono
    if waveform.dim() > 1 and waveform.shape[0] > 1:
        waveform = waveform[:1, ...]

    # Trim too long utterances...
    if APPLY_TRIMMING:
        waveform, sample_rate = apply_trim(waveform, sample_rate)

    # ... or pad too short ones.
    if APPLY_PADDING:
        waveform = apply_pad(waveform, FRAMES_NUMBER)

    return waveform, sample_rate


def resample_wave(waveform, sample_rate, target_sample_rate):
    waveform, sample_rate = torchaudio.sox_effects.apply_effects_tensor(
        waveform, sample_rate, [["rate", f"{target_sample_rate}"]]
    )
    return waveform, sample_rate


def resample_file(path, target_sample_rate, normalize=True):
    waveform, sample_rate = torchaudio.sox_effects.apply_effects_file(
        path, [["rate", f"{target_sample_rate}"]], normalize=normalize
    )

    return waveform, sample_rate


def apply_trim(waveform, sample_rate):
    (
        waveform_trimmed,
        sample_rate_trimmed,
    ) = torchaudio.sox_effects.apply_effects_tensor(waveform, sample_rate, SOX_SILENCE)

    if waveform_trimmed.size()[1] > 0:
        waveform = waveform_trimmed
        sample_rate = sample_rate_trimmed

    return waveform, sample_rate


def apply_pad(waveform, cut):
    """Pad wave by repeating signal until `cut` length is achieved."""
    waveform = waveform.squeeze(0)
    waveform_len = waveform.shape[0]

    if waveform_len >= cut:
        return waveform[:cut]

    # need to pad
    num_repeats = int(cut / waveform_len) + 1
    padded_waveform = torch.tile(waveform, (1, num_repeats))[:, :cut][0]

    return padded_waveform

# class jddContextDataset(BaseAudioFakeDataset):
#     def __init__(
#         self,
#         root_dir,
#         subset=None,
#         normalize: bool = True,
#         meta_name='meta.csv',
#     ):
#         super().__init__(subset=subset)
#         self.root_dir = root_dir
#         self.meta_name = meta_name
#
#         self.read_samples()
#
#         if normalize:
#             all_data = [[] for _ in range(6)]
#
#             for path_c in self.samples["path_context"]:
#                 with open(path_c) as f:
#                     context = json.load(f)[0]
#
#                 wikidata = context["wikidata"]["results"]
#                 all_data[0].append(wikidata["followers_mean"])
#                 all_data[1].append(wikidata["followers_sum"])
#                 all_data[2].append(wikidata["spouse_count"])
#                 all_data[3].append(wikidata["children_count"])
#                 all_data[4].append(len(context["news"]["results"]))
#                 all_data[5].append(len(context["reddit"]["results"]))
#
#             self.c_avg = [np.average(data) for data in all_data]
#             self.c_std = [np.std(data) for data in all_data]
#         else:
#             # default values
#             self.c_avg = [14744176.33995338, 122235791.67118645, 1.159322033898305, 1.8101694915254238, 9.00677966101695, 9.99322033898305]
#             self.c_std = [25298336.46708015, 247163191.71261436, 0.9699062761059759, 2.361678153365988, 2.8617799609108783, 0.11624697084394742]
#
#     def read_samples(self):
#         path = Path(self.root_dir)
#         meta_path = path / self.meta_name
#
#         self.samples = pd.read_csv(meta_path)
#         self.samples["path"] = self.samples["file_name"].apply(lambda n: str(path / "dataset_audio" / f"{n}.wav"))
#         # self.samples["path"] = self.samples["file_name"].apply(lambda n: str(path / "data" / f"{n}.wav"))
#         self.samples["path_transcript"] = self.samples["file_name"].apply(lambda n: str(path / "transcripts_embeddings" / f"{n}.pt"))
#         self.samples["path_context"] = self.samples["file_name"].apply(lambda n: str(path / "context" / f"{n}.json"))
#         self.samples["label"] = self.samples["label"].map({"real": "bonafide", "fake": "spoof"})
#
#         # Keep only the samples for the specified subset
#         if self.subset is not None:
#             self.samples = self.samples[self.samples["split"] == self.subset]
#
#     def __getitem__(self, index) -> T_co:
#         waveform, label = super().__getitem__(index)
#         sample = self.samples.iloc[index]
#
#         # transcript
#         transcript_embedding = torch.load(sample["path_transcript"], map_location=torch.device('cpu')).float()
#
#         with open(sample["path_context"]) as f:
#             context = json.load(f)
#
#         context = context[0]
#         context_embedding = torch.Tensor([
#             context["wikidata"]["results"]["followers_mean"],
#             context["wikidata"]["results"]["followers_sum"],
#             context["wikidata"]["results"]["spouse_count"],
#             context["wikidata"]["results"]["children_count"],
#             len(context["news"]["results"]),
#             len(context["reddit"]["results"])
#         ])
#
#         # (a - avg) / std
#         context_embedding = torch.div((context_embedding - torch.Tensor(self.c_avg)), torch.Tensor(self.c_std))
#
#         return (waveform, transcript_embedding, context_embedding), label