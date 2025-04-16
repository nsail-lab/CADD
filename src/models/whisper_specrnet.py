import numpy as np
import torch

from src import frontends
from src.models.whisper_main import ModelDimensions, Whisper, log_mel_spectrogram
from src.models.specrnet import SpecRNet, SpecRNetContext
from src.util import WHISPER_MODEL_WEIGHTS_PATH


class WhisperSpecRNet(SpecRNet):
    def __init__(self, input_channels, freeze_encoder, **kwargs):
        super().__init__(input_channels=input_channels, **kwargs)

        self.device = kwargs["device"]
        checkpoint = torch.load(WHISPER_MODEL_WEIGHTS_PATH)
        dims = ModelDimensions(**checkpoint["dims"].__dict__)
        model = Whisper(dims)
        model = model.to(self.device)
        model.load_state_dict(checkpoint["model_state_dict"])
        self.whisper_model = model
        if freeze_encoder:
            for param in self.whisper_model.parameters():
                param.requires_grad = False

    def compute_whisper_features(self, x):
        specs = []
        for sample in x:
            specs.append(log_mel_spectrogram(sample))
        x = torch.stack(specs)
        x = self.whisper_model(x)

        x = x.permute(0, 2, 1)  # (bs, frames, 3 x n_lfcc)
        x = x.unsqueeze(1)  # (bs, 1, frames, 3 x n_lfcc)
        x = x.repeat(
            (1, 1, 1, 2)
        )  # (bs, 1, frames, 3 x n_lfcc) -> (bs, 1, frames, 3000)
        return x

    def forward(self, x):
        # we assume that the data is correct (i.e. 30s)
        x = self.compute_whisper_features(x)
        out = self._compute_embedding(x)
        return out

class WhisperSpecRNetContext(SpecRNetContext):
    def __init__(self, input_channels, freeze_encoder, **kwargs):
        super().__init__(input_channels=input_channels, **kwargs)

        self.device = kwargs["device"]
        checkpoint = torch.load(WHISPER_MODEL_WEIGHTS_PATH)
        dims = ModelDimensions(**checkpoint["dims"].__dict__)
        model = Whisper(dims)
        model = model.to(self.device)
        model.load_state_dict(checkpoint["model_state_dict"])
        self.whisper_model = model
        if freeze_encoder:
            for param in self.whisper_model.parameters():
                param.requires_grad = False

    def compute_whisper_features(self, x):
        specs = []
        for sample in x:
            specs.append(log_mel_spectrogram(sample))
        x = torch.stack(specs)
        x = self.whisper_model(x)

        x = x.permute(0, 2, 1)  # (bs, frames, 3 x n_lfcc)
        x = x.unsqueeze(1)  # (bs, 1, frames, 3 x n_lfcc)
        x = x.repeat(
            (1, 1, 1, 2)
        )  # (bs, 1, frames, 3 x n_lfcc) -> (bs, 1, frames, 3000)
        return x

    def forward(self, x, c_e):
        # we assume that the data is correct (i.e. 30s)
        x = self.compute_whisper_features(x)
        out = self._compute_embedding(x, c_e)
        return out

class WhisperMultiFrontSpecRNet(WhisperSpecRNet):
    def __init__(self, input_channels, freeze_encoder, **kwargs):
        super().__init__(
            input_channels=input_channels,
            freeze_encoder=freeze_encoder,
            **kwargs,
        )

        self.frontend = frontends.get_frontend(kwargs['frontend_algorithm'], kwargs['device'])

    def forward(self, x):
        frontend_x = self.frontend(x)
        x = self.compute_whisper_features(x)

        x = torch.cat([x, frontend_x], 1)
        out = self._compute_embedding(x)
        return out

class WhisperMultiFrontSpecRNet_lfccmfcc(WhisperSpecRNet):
    def __init__(self, input_channels, freeze_encoder, **kwargs):
        super().__init__(
            input_channels=input_channels,
            freeze_encoder=freeze_encoder,
            **kwargs,
        )

        frontend_name = kwargs.get("frontend_algorithm", [])
        self.frontend_1 = frontends.get_frontend(frontend_name[0], self.device)
        self.frontend_2 = frontends.get_frontend(frontend_name[1], self.device)
        print(f"Using {frontend_name} frontend")

    def forward(self, x):
        frontend_x_1 = self.frontend_1(x)
        frontend_x_2 = self.frontend_2(x)
        x = self.compute_whisper_features(x)

        x = torch.cat([x, frontend_x_1, frontend_x_2], 1)
        out = self._compute_embedding(x)
        return out

class WhisperMultiFrontSpecRNetContext(WhisperSpecRNetContext):
    def __init__(self, input_channels, freeze_encoder, **kwargs):
        super().__init__(
            input_channels=input_channels,
            freeze_encoder=freeze_encoder,
            **kwargs,
        )

        self.frontend = frontends.get_frontend(kwargs['frontend_algorithm'], kwargs['device'])

    def forward(self, x, c_e):
        frontend_x = self.frontend(x)
        x = self.compute_whisper_features(x)

        x = torch.cat([x, frontend_x], 1)
        out = self._compute_embedding(x, c_e)
        return out

class WhisperMultiFrontSpecRNetContext_lfccmfcc(WhisperSpecRNetContext):
    def __init__(self, input_channels, freeze_encoder, **kwargs):
        super().__init__(
            input_channels=input_channels,
            freeze_encoder=freeze_encoder,
            **kwargs,
        )

        frontend_name = kwargs.get("frontend_algorithm", [])
        self.frontend_1 = frontends.get_frontend(frontend_name[0], self.device)
        self.frontend_2 = frontends.get_frontend(frontend_name[1], self.device)
        print(f"Using {frontend_name} frontend")

    def forward(self, x, c_e):
        frontend_x_1 = self.frontend_1(x)
        frontend_x_2 = self.frontend_2(x)
        x = self.compute_whisper_features(x)

        x = torch.cat([x, frontend_x_1, frontend_x_2], 1)
        out = self._compute_embedding(x, c_e)
        return out