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
    captured["intermediate_dense_input"] = args[0]

def capture_h(module, args, result):
    captured["output_dense_output"] = result

input_hook  = layer0.intermediate.dense.register_forward_hook(capture_x)
output_hook = layer0.output.dense.register_forward_hook(capture_h)

#capture
with torch.inference_mode():
    output = model(**inputs)

#release
input_hook.remove()
output_hook.remove()

#validate
print("FFN input: ", captured["intermediate_dense_input"].shape)
print("FFN output:", captured["output_dense_output"].shape)
with torch.inference_mode():
    intermediate_output = layer0.intermediate(captured["intermediate_dense_input"])
    output_dense_output = layer0.output.dense(intermediate_output)

torch.testing.assert_close(output_dense_output, captured["output_dense_output"])
print("match!")