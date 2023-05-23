import torch
import torch.nn as nn
from transformers import AutoModelForTokenClassification
from transformers import BertConfig

class BERTSeg(nn.Module):
    def __init__(self, num_classes, num_features) -> None:
        super().__init__()
        config = BertConfig.from_json_file("bert-tiny.json")
        self.inp = nn.Linear(num_features, 128)
        self.encoder = AutoModelForTokenClassification.from_config(config)
        self.encoder.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.inp(x)
        return self.encoder(inputs_embeds=x)