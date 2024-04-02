import os

from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM

HUGGINGFACE_TOKEN = os.environ.get("HUGGINGFACE_TOKEN")
login(token=HUGGINGFACE_TOKEN)

tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b-it")
model = AutoModelForCausalLM.from_pretrained("google/gemma-2b-it")

input_text = "List ten plans for tourists in Malaga, Spain."
input_ids = tokenizer(input_text, return_tensors="pt")

outputs = model.generate(**input_ids, max_length=512)
print(tokenizer.decode(outputs[0]))
