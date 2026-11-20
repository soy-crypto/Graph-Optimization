from pathlib import Path
from typing import cast

from transformers import AutoTokenizer
from transformers.models.bert.modeling_bert import BertModel, BertLayer

import torch
import 
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
def capture_ffn_input(module, args, result):
    captured["intermediate_dense_input"] = args[0]

def capture_ffn_output(module, args, result):
    captured["output_dense_output"] = result

input_hook  = layer0.intermediate.dense.register_forward_hook(capture_ffn_input)
output_hook = layer0.output.dense.register_forward_hook(capture_ffn_output)

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
print("Recomputed layer0.output.dense output matches the captured output.")

#save
ffn_first_module = layer0.intermediate.dense
ffn_last_module  = layer0.output.dense
path = Path("build/bert-tiny-layer0-ffn.npz")
np.savez(path,
         X=captured["intermediate_dense_input"].numpy(),
         W1=ffn_first_module.weight.detach().numpy(),
         bias1=ffn_first_module.bias.detach().numpy(),
         W2=ffn_last_module.weight.detach().numpy(),
         bias2=ffn_last_module.bias.detach().numpy(),
         Y=captured["output_dense_output"].numpy()
)

with np.load(path) as saved:
    for name in saved.files:
        print(name, saved[name].shape)
        
#ffn computation graph
ffn = torch.nn.Sequential(layer0.intermediate, layer0.output.dense).eval()
with np.load(path) as saved:
    X = torch.from_numpy(saved["X"])

computation_graph = torch.export.export(ffn, (X,))
print(computation_graph.graph_module.code)