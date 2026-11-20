from transformers import AutoModel
from transformers import AutoTokenizer

model = AutoModel.from_pretrained("google/bert_uncased_L-2_H-128_A-2")
tokenizer = AutoTokenizer.from_pretrained("google/bert_uncased_L-2_H-128_A-2")

model.eval()
print(type(model).__name__)

print("Hidden size:", model.config.hidden_size)
print("Intermeidate size:", model.config.intermediate_size)
print(tokenizer.tokenize("I like cats."))