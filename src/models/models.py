from typing import Dict
from src.models import (
    lcnn,
    specrnet,
    whisper_specrnet,
    rawnet3,
    whisper_lcnn,
    meso_net,
    whisper_meso_net
)


def get_model(model_name: str, config: Dict, device: str):
    if model_name == "rawnet3":
        return rawnet3.prepare_model()
    elif model_name == "lcnn":
        return lcnn.FrontendLCNN(device=device, **config)
    elif model_name == "mesonet":
        return meso_net.FrontendMesoInception4(
            input_channels=config.get("input_channels", 1),
            fc1_dim=config.get("fc1_dim", 1024),
            frontend_algorithm=config.get("frontend_algorithm", "lfcc"),
            device=device,
        )
    elif model_name == "specrnet":
        return specrnet.FrontendSpecRNet(
            device=device,
            **config,
        )
    elif model_name == "lfcc_mfcc_lcnn":
        return lcnn.FrontendLCNN_lfccmfcc(device=device, **config)
    elif model_name == "lfcc_mfcc_mesonet":
        return meso_net.FrontendMesoInception4_lfccmfcc(
            input_channels=config.get("input_channels", 1),
            fc1_dim=config.get("fc1_dim", 1024),
            frontend_algorithm=config.get("frontend_algorithm", "lfcc"),
            device=device,
        )
    elif model_name == "rawnet3_context":
        return rawnet3.prepare_model_context(
            embedding_dim=config.get("embedding_dim", 100),
            context_jdd=config.get("context", "ct"),
        )
    elif model_name == "lcnn_context":
        return lcnn.FrontendLCNNContext(
            # embedding_dim=config.get("embedding_dim", 100),
            # context_jdd=config.get("context", "ct"),
            device=device,
            **config)
    elif model_name == "mesonet_context":
        return meso_net.FrontendMesoContext(
            input_channels=config.get("input_channels", 1),
            fc1_dim=config.get("fc1_dim", 1024),
            frontend_algorithm=config.get("frontend_algorithm", "lfcc"),
            embedding_dim=config.get("embedding_dim", 100),
            device=device,
            context=config.get("context", "ct"),
        )
    elif model_name == "specrnet_context":
        return specrnet.FrontendSpecRNetContext(
            input_channels=config.get("input_channels", 1),
            frontend_algorithm=config.get("frontend_algorithm", "lfcc"),
            embedding_dim=config.get("embedding_dim", 100),
            device=device,
            context=config.get("context", "ct"),
        )
    elif model_name == "lfcc_mfcc_lcnn_context":
        return lcnn.FrontendLCNNContext_lfccmfcc(
            # embedding_dim=config.get("embedding_dim", 100),
            # context_jdd=config.get("context", "ct"),
            device=device,
            **config)
    elif model_name == "lfcc_mfcc_mesonet_context":
        return meso_net.FrontendMesoContext_lfccmfcc(
            input_channels=config.get("input_channels", 1),
            fc1_dim=config.get("fc1_dim", 1024),
            frontend_algorithm=config.get("frontend_algorithm", "lfcc"),
            embedding_dim=config.get("embedding_dim", 100),
            device=device,
            context=config.get("context", "ct"),
        )
    elif model_name == "lfcc_mfcc_specrnet":
        return specrnet.FrontendSpecRNet_lfccmfcc(
            device=device,
            **config,
        )
    elif model_name == "lfcc_mfcc_specrnet_context":
        return specrnet.FrontendSpecRNetContext_lfccmfcc(
            input_channels=config.get("input_channels", 1),
            frontend_algorithm=config.get("frontend_algorithm", "lfcc"),
            embedding_dim=config.get("embedding_dim", 100),
            device=device,
            context=config.get("context", "ct"),
        )
    elif model_name == "whisper_lcnn":
        return whisper_lcnn.WhisperLCNN(
            input_channels=config.get("input_channels", 1),
            freeze_encoder=config.get("freeze_encoder", False),
            device=device,
        )
    elif model_name == "whisper_mesonet":
        return whisper_meso_net.WhisperMesoNet(
            input_channels=config.get("input_channels", 1),
            freeze_encoder=config.get("freeze_encoder", False),
            fc1_dim=config.get("fc1_dim", 1024),
            device=device,
        )
    elif model_name == "whisper_specrnet":
        return whisper_specrnet.WhisperSpecRNet(
            input_channels=config.get("input_channels", 1),
            freeze_encoder=config.get("freeze_encoder", False),
            device=device,
        )
    elif model_name == "whisper_lcnn_context":
        return whisper_lcnn.WhisperLCNNContext(
            input_channels=config.get("input_channels", 1),
            freeze_encoder=config.get("freeze_encoder", False),
            embedding_dim=config.get("embedding_dim", 100),
            device=device,
            context=config.get("context", "ct"),
        )
    elif model_name == "whisper_mesonet_context":
        return whisper_meso_net.WhisperMesoContext(
            input_channels=config.get("input_channels", 1),
            freeze_encoder=config.get("freeze_encoder", False),
            fc1_dim=config.get("fc1_dim", 1024),
            embedding_dim=config.get("embedding_dim", 100),
            device=device,
            context=config.get("context", "ct"),
        )
    elif model_name == "whisper_specrnet_context":
        return whisper_specrnet.WhisperSpecRNetContext(
            input_channels=config.get("input_channels", 1),
            freeze_encoder=config.get("freeze_encoder", False),
            embedding_dim=config.get("embedding_dim", 100),
            device=device,
            context=config.get("context", "ct"),
        )
    elif model_name == "whisper_frontend_lcnn":
        return whisper_lcnn.WhisperMultiFrontLCNN(
            input_channels=config.get("input_channels", 2),
            freeze_encoder=config.get("freeze_encoder", False),
            frontend_algorithm=config.get("frontend_algorithm", "lfcc"),
            device=device,
        )
    elif model_name == "whisper_frontend_mesonet":
        return whisper_meso_net.WhisperMultiFrontMesoNet(
            input_channels=config.get("input_channels", 2),
            fc1_dim=config.get("fc1_dim", 1024),
            freeze_encoder=config.get("freeze_encoder", True),
            frontend_algorithm=config.get("frontend_algorithm", "lfcc"),
            device=device,
        )
    elif model_name == "whisper_frontend_specrnet":
        return whisper_specrnet.WhisperMultiFrontSpecRNet(
            input_channels=config.get("input_channels", 2),
            freeze_encoder=config.get("freeze_encoder", False),
            frontend_algorithm=config.get("frontend_algorithm", "lfcc"),
            device=device,
        )
    elif model_name == "whisper_lfcc_mfcc_frontend_lcnn":
        return whisper_lcnn.WhisperMultiFrontLCNN_lfccmfcc(
            input_channels=config.get("input_channels", 2),
            freeze_encoder=config.get("freeze_encoder", False),
            frontend_algorithm=config.get("frontend_algorithm", "lfcc"),
            device=device,
        )
    elif model_name == "whisper_lfcc_mfcc_frontend_mesonet":
        return whisper_meso_net.WhisperMultiFrontMesoNet_lfccmfcc(
            input_channels=config.get("input_channels", 2),
            fc1_dim=config.get("fc1_dim", 1024),
            freeze_encoder=config.get("freeze_encoder", True),
            frontend_algorithm=config.get("frontend_algorithm", "lfcc"),
            device=device,
        )
    elif model_name == "whisper_lfcc_mfcc_frontend_specrnet":
        return whisper_specrnet.WhisperMultiFrontSpecRNet_lfccmfcc(
            input_channels=config.get("input_channels", 2),
            freeze_encoder=config.get("freeze_encoder", False),
            frontend_algorithm=config.get("frontend_algorithm", "lfcc"),
            device=device,
        )
    elif model_name == "whisper_frontend_lcnn_context":
        return whisper_lcnn.WhisperMultiFrontLCNNContext(
            input_channels=config.get("input_channels", 2),
            freeze_encoder=config.get("freeze_encoder", False),
            frontend_algorithm=config.get("frontend_algorithm", "lfcc"),
            embedding_dim=config.get("embedding_dim", 100),
            device=device,
            context=config.get("context", "ct"),
        )
    elif model_name == "whisper_frontend_mesonet_context":
        return whisper_meso_net.WhisperMultiFrontMesoContext(
            input_channels=config.get("input_channels", 2),
            freeze_encoder=config.get("freeze_encoder", False),
            fc1_dim=config.get("fc1_dim", 1024),
            frontend_algorithm=config.get("frontend_algorithm", "lfcc"),
            embedding_dim=config.get("embedding_dim", 100),
            device=device,
            context=config.get("context", "ct"),
        )
    elif model_name == "whisper_frontend_specrnet_context":
        return whisper_specrnet.WhisperMultiFrontSpecRNetContext(
            input_channels=config.get("input_channels", 2),
            freeze_encoder=config.get("freeze_encoder", False),
            frontend_algorithm=config.get("frontend_algorithm", "lfcc"),
            embedding_dim=config.get("embedding_dim", 100),
            device=device,
            context=config.get("context", "ct"),
        )
    elif model_name == "whisper_lfcc_mfcc_frontend_lcnn_context":
        return whisper_lcnn.WhisperMultiFrontLCNNContext_lfccmfcc(
            input_channels=config.get("input_channels", 2),
            freeze_encoder=config.get("freeze_encoder", False),
            frontend_algorithm=config.get("frontend_algorithm", "lfcc"),
            embedding_dim=config.get("embedding_dim", 100),
            device=device,
            context=config.get("context", "ct"),
        )
    elif model_name == "whisper_lfcc_mfcc_frontend_mesonet_context":
        return whisper_meso_net.WhisperMultiFrontMesoContext_lfccmfcc(
            input_channels=config.get("input_channels", 2),
            freeze_encoder=config.get("freeze_encoder", False),
            fc1_dim=config.get("fc1_dim", 1024),
            frontend_algorithm=config.get("frontend_algorithm", "lfcc"),
            embedding_dim=config.get("embedding_dim", 100),
            device=device,
            context=config.get("context", "ct"),
        )
    elif model_name == "whisper_lfcc_mfcc_frontend_specrnet_context":
        return whisper_specrnet.WhisperMultiFrontSpecRNetContext_lfccmfcc(
            input_channels=config.get("input_channels", 2),
            freeze_encoder=config.get("freeze_encoder", False),
            frontend_algorithm=config.get("frontend_algorithm", "lfcc"),
            embedding_dim=config.get("embedding_dim", 100),
            device=device,
            context=config.get("context", "ct"),
        )
    else:
        raise ValueError(f"Model '{model_name}' not supported")