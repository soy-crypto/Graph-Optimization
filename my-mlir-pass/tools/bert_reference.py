from pathlib import Path

from transformers import AutoModel, AutoTokenizer
from transformers.models.bert.modeling_bert import BertModel
import torch
import numpy as np

#init
model_name = "google/bert_uncased_L-2_H-128_A-2"
model      = BertModel.from_pretrained(model_name)
assert isinstance(model, BertModel)
tokenizer  = AutoTokenizer.from_pretrained(model_name)
model.eval()

print("layer0", model.encoder.layer[0])


'''
#
print(type(model).__name__)
print("Hidden size:", model.config.hidden_size)
print("Intermeidate size:", model.config.intermediate_size)

#capture
captured = {}
def capture_input(module, args):
    captured["ffn_input"] = args[0]

handle = layer.intermediate.dense.register_forward_pre_hook(capture_input)

#inference
with torch.inference_mode():
    output = model(**inputs)

handle.remove()
print("First FFN input:", captured["ffn_input"].shape)
print("Output shape:", output.lat_hidden_state.shape)

#save inputs and output
reference = {}
for name, tensor in inputs.items():
    reference[name] = tensor.numpy()

reference["ffn_input"]         = captured["ffn_input"].numpy()
reference["last_hidden_state"] = output.last_hidden_state.numpy()

path = Path("build/bert-tiny-reference.npz")
np.savez(path, **reference)

#read
with np.load(path) as saved:
    print("saved output shape:", saved["last_hidden_state"].shape)
    
    
#request one sequence with 64 positions
inputs = tokenizer("I like cats", padding="max_length", max_length=64, return_tensors="pt")
print("Input name:", list(inputs))
print("Input shaep:", inputs["input_ids"].shape)
print("Active tokens:", inputs["attention_mask"].sum().item())

#inspect the first encoder block's feed-forward network
layer0 = model.encoder.layer[0]
print(layer0)
for name, module in layer0.named_modules():
    print(name, type(module).__name__)



print("First linear layer:", layer0.intermediate.dense)
print("Second linear layer:", layer0.output.dense)
'''

