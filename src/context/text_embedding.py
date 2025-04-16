import torch
from transformers import (
    AlbertModel, 
    AutoTokenizer, 
    WhisperProcessor, 
    WhisperForConditionalGeneration,
    PegasusForConditionalGeneration, 
    PegasusTokenizer
)

class TextEmbedder:
    """Generates text embeddings (defaults to using ALBERT model)."""
    def __init__(self, model_name_or_path='sentence-transformers/paraphrase-albert-small-v2', device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = torch.device(device)
        self.albert_token = AutoTokenizer.from_pretrained(model_name_or_path)
        self.albert_model = AlbertModel.from_pretrained(model_name_or_path).to(device)
    
    def embed(self, text):
        encoded_input = self.albert_token(text, padding=True, truncation=True, return_tensors='pt').to(self.device)
        with torch.no_grad():
            model_output = self.albert_model(**encoded_input)
            sentence_embeddings = self.__mean_pooling(model_output, encoded_input['attention_mask'])
        return sentence_embeddings
    
    def __mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
class SpeechToText:
    """Transcribes text from speech (defaults to using Whisper model)."""
    def __init__(self, model_name_or_path='openai/whisper-large-v2', language='en', device="cuda:0" if torch.cuda.is_available() else "cpu"):
        self.model_name_or_path = model_name_or_path
        self.processor = WhisperProcessor.from_pretrained(model_name_or_path)
        self.asr_model = WhisperForConditionalGeneration.from_pretrained(model_name_or_path)
        self.asr_model.config.forced_decoder_ids = None
        self.asr_model.generation_config.language = language
        self.device = torch.device(device)
        self.asr_model.to(self.device)
    
    def transcribe(self, signal, sampling_rate):
        input_features = self.processor(signal, sampling_rate=sampling_rate, return_tensors="pt").input_features.to(self.device)
        predicted_ids = self.asr_model.generate(input_features)
        transcription = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)
        return transcription

class TextSummarizer:
    """Summarizes text (defaults to using Pegasus model)."""
    def __init__(self, model_name_or_path='google/pegasus-xsum', device="cuda:0" if torch.cuda.is_available() else "cpu"):
        self.device = torch.device(device)
        self.pegasus_token = PegasusTokenizer.from_pretrained(model_name_or_path)
        self.pegasus_model = PegasusForConditionalGeneration.from_pretrained(model_name_or_path).to(device)
    
    def summarize(self, text):
        batch = self.pegasus_token(text, truncation=True, padding='longest', return_tensors="pt").to(self.device)
        with torch.no_grad():
            translated = self.pegasus_model.generate(**batch)
            tgt_text = self.pegasus_token.batch_decode(translated, skip_special_tokens=True)
        return tgt_text
