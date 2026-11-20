from pathlib import Path

from transformers import AutoModel, AutoTokenizer
import torch
import numpy as np

#init
model_name = "google/bert_uncased_L-2_H-128_A-2"
model = AutoModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model.eval()

print(type(model).__name__)
print("Hidden size:", model.config.hidden_size)
print("Intermeidate size:", model.config.intermediate_size)

#inpect the first encoder block's feed-forward network
layer = model.encoder.layer[0]
print("First linear layer:", layer.intermediate.dense)
print("Second linear layer:", layer.output.dense)

#request one sequence with 64 positions
inputs = tokenizer("I like cats", padding="max_length", max_length=64, return_tensors="pt")
print("Input name:", list(inputs))
print("Input shaep:", inputs["input_ids"].shape)
print("Active tokens:", inputs["attention_mask"].sum().item())

#
captured = {}
def capture_input():
    captured["ffn_input"] = args[0]

handle = layer.intermediate.dense.register_forward_pre_hook(capture_input)

#inference
with torch.inference_mode():
    output = model(**inputs)

print("Output shape:", output.last_hidden_state.shape)
handle.remove()
print("First FFN input:", captured["ffn_input"].shape)
print("Output shape:", output.lat_hidden_state.shape)

#save inputs and output
reference = {}
for name, tensor in inputs.items():
    reference[name] = tensor.numpy()

reference["last_hidden_state"] = output.last_hidden_state.numpy()
path = Path("build/bert-tiny-reference.npz")
np.savez(path, **reference)

#read
with np.load(path) as saved:
    print("saved output shape:", saved["last_hidden_state"].shape)