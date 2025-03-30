# Llama-3-fine_tuning
# LLaMA 3.2-3B Fine-Tuning with FineTome-100k and Unsloth

## 🚀 Project Overview
This project fine-tunes the `llama-3.2-3B-instruct` model using the `FineTome-100k` dataset to enhance instruction-following capabilities. To accelerate training, we leverage [Unsloth](https://github.com/unslothai/unsloth), a highly optimized library designed to speed up LLaMA-based fine-tuning while maintaining efficiency and scalability.

## 📌 Dataset: FineTome-100k
- **Source:** [`mlabonne/FineTome-100k`](https://huggingface.co/datasets/mlabonne/FineTome-100k)
- **Size:** 100,000 high-quality instruction-response pairs
- **Purpose:** The dataset consists of structured instruction-response data designed to improve the generalization of instruction-tuned LLMs.
- **Use Case:** Enhances models in structured response generation, instruction following, and task-specific completions.

## 🛠️ Setup & Installation
### 1️⃣ Install Required Dependencies
```bash
pip install torch transformers datasets accelerate unsloth bitsandbytes
```
### 2️⃣ Load the Dataset
```python
from datasets import load_dataset

dataset = load_dataset("mlabonne/FineTome-100k", split="train")
print(dataset[0])  # Check first sample
```

### 3️⃣ Load LLaMA 3.2-3B with Unsloth
```python
from unsloth import FastLlamaForCausalLM, FastLlamaTokenizer

model_name = "meta-llama/Meta-Llama-3-3B-Instruct"
model, tokenizer = FastLlamaForCausalLM.from_pretrained(model_name, load_in_8bit=True)
```

### 4️⃣ Preprocess the Data
```python
def tokenize_function(example):
    return tokenizer(example["instruction"], example["response"], truncation=True, padding="max_length")

tokenized_dataset = dataset.map(tokenize_function, batched=True)
```

### 5️⃣ Fine-Tune the Model with PEFT & LoRA
```python
from peft import LoraConfig, get_peft_model

config = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.1)
model = get_peft_model(model, config)

# Training setup
from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
)

trainer.train()
```

## ⏩ Why Use Unsloth?
✅ **Faster Training** – Optimized for LLaMA models, reducing training time significantly.  
✅ **Efficient Memory Usage** – Supports quantization (4-bit, 8-bit) to fit large models into limited resources.  
✅ **Seamless Integration** – Works effortlessly with Hugging Face Transformers and PEFT for LoRA fine-tuning.

## 📈 Evaluation
Once trained, you can evaluate the model:
```python
prompt = "Explain quantum computing in simple terms."
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs)
print(tokenizer.decode(outputs[0]))
```

## 🚀 Conclusion
This project demonstrates efficient fine-tuning of `llama-3.2-3B-instruct` using `FineTome-100k` and `Unsloth`, making instruction-tuned LLMs faster and more effective.

---

📌 **Contributors**: R.Sarath Kumar 
📌 **License**: MIT  
📌 **References**: Meta AI, Hugging Face, Unsloth

