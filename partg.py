from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
import torch
from torch.utils.data import Dataset

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
model = GPT2LMHeadModel.from_pretrained('results/checkpoint-18897')
model.resize_token_embeddings(len(tokenizer))
def startgpt():
    return tokenizer, model
# Пример данных для дообучения (ваш датасет)
with open('traning', 'r') as f:
    t = f.readlines()
train_texts = ["1"]


class CustomDataset(Dataset):
    def __init__(self, tokenizer, texts):
        self.tokenizer = tokenizer
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(self.texts[idx], truncation=True, padding='max_length', max_length=128,
                                  return_tensors='pt')
        item = {key: val.squeeze() for key, val in encoding.items()}
        item['labels'] = item['input_ids'].clone()
        return item


train_dataset = CustomDataset(tokenizer, train_texts)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Импорт Trainer и TrainingArguments
training_args = TrainingArguments(
    output_dir='./results',  # Директория для сохранения модели и логов
    per_device_train_batch_size=4,  # Размер батча на одно устройство
    num_train_epochs=1000,  # Количество эпох обучения
    logging_dir='./logs',  # Директория для логов
)

trainer = Trainer(
    model=model,  # Модель для дообучения
    args=training_args,  # Параметры обучения
    train_dataset=train_dataset,  # Датасет для обучения
    data_collator=data_collator,  # Collator для добавления padding
)

trainer.train()

# Сохранение дообученной модели
model.save_pretrained("./gpt2-finetuned")

# Пример генерации текста с дообученной моделью
def gen(text, tokenizer, model):
    input_text = text
    input_ids = tokenizer.encode(input_text, return_tensors='pt')
    output = model.generate(input_ids, max_length=300, num_beams=8, no_repeat_ngram_size=2, do_sample=True, early_stopping=False, eos_token_id=50256, top_k=50)
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text


t, m = startgpt()
print(gen("hi", t, m))