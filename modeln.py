import torch
import torch.nn as nn
from transformers import AutoModelForTokenClassification
from transformers import BertConfig

class PositionalEmbedding(nn.Module):
    def __init__(self, embedding_dim, max_position=512):
        super(PositionalEmbedding, self).__init__()
        self.position_embeddings = nn.Embedding(max_position, embedding_dim)

    def forward(self, inputs):
        seq_length = inputs.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=inputs.device)
        position_embeddings = self.position_embeddings(position_ids)
        return inputs + position_embeddings
    
class BERTSegm(nn.Module):
    def __init__(self, num_classes, num_features) -> None:
        super().__init__()
        config = BertConfig.from_json_file("bert-tiny.json")
        self.inp = nn.Linear(num_features, 128)
        self.encoder = AutoModelForTokenClassification.from_config(config)
        self.classifier = nn.Linear(128, num_classes)
        self.pos_embed = PositionalEmbedding(128)

    def forward(self, x):
        x = self.inp(x)
        x = self.pos_embed(x)
        out = self.encoder.bert.encoder(x)
        return self.classifier(out.last_hidden_state)