from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "meta-llama/Llama-2-13b-chat-hf"
# model_name = "meta-llama/Llama-2-13b-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

tokenizer.add_special_tokens({'pad_token': '[PAD]'})
# model.resize_token_embeddings(len(tokenizer))  # Adjust model embeddings to account for new token
model.resize_token_embeddings(len(tokenizer), mean_resizing=False)


context = """
The LLaMA models are state-of-the-art language models that have achieved impressive results in several NLP tasks.
They are designed to scale to larger data sets and provide more accurate language understanding and generation capabilities.
"""
question = "What are LLaMA models known for?"

# Prepare the input as a concatenation of context and question
input_text = f"Context: {context}\nQuestion: {question}\nAnswer:"

inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True, max_length=1024)
input_ids = inputs['input_ids']

# Move input_ids to the correct device (GPU in this case)
input_ids = inputs['input_ids'].to(model.device)

# Generate answer
# answer_output = model.generate(**inputs, max_length=100, num_return_sequences=1, do_sample=False)
# answer_output = model.generate(input_ids, do_sample=True, temperature=0.6)
answer_output = model.generate(input_ids, max_length=200, num_return_sequences=1, do_sample=False)

answer = tokenizer.decode(answer_output[0], skip_special_tokens=True)
print("Answer:", answer)
