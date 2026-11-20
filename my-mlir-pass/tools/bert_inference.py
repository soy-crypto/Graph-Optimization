from transformers import AutoModel, AutoTokenizer

model     = AutoModel.from_pretrained("google/bert_uncased_L-2_H-128_A-2")
tokenizer = AutoTokenizer.from_pretrained("google/bert_uncased_L-2_H-128_A-2")
model.eval()

print(type(model).__name__)
print("Hidden size:", model.config.hidden_size)
print("Intermeidate size:", model.config.intermediate_size)
print(tokenizer.tokenize("I like cats."))
print(tokenizer.tokenize("catlike"))

inputs = tokenizer("I like cats", return_tensors="pt")
print(inputs["input_ids"].shape)
print(tokenizer.convert_ids_to_tokens(inputs["input_ids"][0].tolist()))