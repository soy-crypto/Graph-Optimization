from pathlib import Path
from typing import cast

from transformers import AutoTokenizer
from transformers.models.bert.modeling_bert import BertModel, BertLayer

import torch
import numpy as np

#init
model_name = "google/bert_uncased_L-2_H-128_A-2"
model      = BertModel.from_pretrained(model_name)
tokenizer  = AutoTokenizer.from_pretrained(model_name)
model.eval()

inputs = tokenizer("I like cats", padding="max_length", max_length=64, return_tensors="pt")
layer0 = cast(BertLayer, model.encoder.layer[0])
captured = {}

#register hooks
def capture_x(module, args, result):
    captured["ffn_input"] = args[0]

def capture_h(module, args, result):
    captured["ffn_output"] = result

input_hook  = layer0.intermediate.dense.register_forward_hook(capture_x)
output_hook = layer0.output.dense.register_forward_hook(capture_h)

#capture
with torch.inference_mode():
    output = model(**inputs)

#release
input_hook.remove()
output_hook.remove()

print("FFN input: ", captured["ffn_input"].shape)
print("FFN output:", captured["ffn_output"].shape)

#
layer0.output.