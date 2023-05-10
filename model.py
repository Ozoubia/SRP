import torch
import torch.nn as nn
from transformers import AutoModelForTokenClassification
from transformers import BertConfig

class BERTSeg(nn.Module):
    def __init__(self, num_classes) -> None:
        super().__init__()
        config = BertConfig.from_json_file("config.json")
        self.encoder = AutoModelForTokenClassification.from_config(config)
        self.encoder.classifier = nn.Linear(768, num_classes)

    def forward(self, x):
         return self.encoder(input_embeds=x)

#demo   
# t = torch.rand((1,4,768))
# m = BERTSeg(5)

# m(t)

# f = 2+3


