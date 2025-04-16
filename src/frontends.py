from typing import List, Union, Callable
import torch
import torchaudio

SAMPLING_RATE = 16_000
WIN_LENGTH = int((25 / 1_000) * SAMPLING_RATE)  # 400
HOP_LENGTH = int((10 / 1_000) * SAMPLING_RATE)  # 160

DELTA_FN = torchaudio.transforms.ComputeDeltas(
    win_length=WIN_LENGTH,
    mode="replicate",
)

def get_frontend(
    frontends: List[str],
    device: Union[torch.device, str],
) -> Callable:
    if "mfcc" in frontends:
        return create_mfcc_processor(device)
    elif "lfcc" in frontends:
        return create_lfcc_processor(device)
    else:
        raise ValueError(f"{frontends} frontend is not supported!")

def create_mfcc_processor(device: Union[torch.device, str]) -> Callable:
    mfcc_fn = torchaudio.transforms.MFCC(
        sample_rate=SAMPLING_RATE,
        n_mfcc=128,
        melkwargs={
            "n_fft": 512,
            "win_length": WIN_LENGTH,
            "hop_length": HOP_LENGTH,
        },
    ).to(device)

    return lambda input: process_features(input, mfcc_fn)

def create_lfcc_processor(device: Union[torch.device, str]) -> Callable:
    lfcc_fn = torchaudio.transforms.LFCC(
        sample_rate=SAMPLING_RATE,
        n_lfcc=128,
        speckwargs={
            "n_fft": 512,
            "win_length": WIN_LENGTH,
            "hop_length": HOP_LENGTH,
        },
    ).to(device)

    return lambda input: process_features(input, lfcc_fn)

def process_features(input: torch.Tensor, feature_fn: Callable) -> torch.Tensor:
    if input.ndim < 4:
        input = input.unsqueeze(1)  # (bs, 1, n_lfcc, frames)
    x = feature_fn(input)
    delta = DELTA_FN(x)
    double_delta = DELTA_FN(delta)
    x = torch.cat((x, delta, double_delta), 2)  # -> [bs, 1, 128 * 3, 1500]
    return x[:, :, :, :3000]  # (bs, n, n_lfcc * 3, frames)