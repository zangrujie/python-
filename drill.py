import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from transformers import Trainer, TrainingArguments
from sklearn.metrics import f1_score, classification_report
import pandas as pd
from tqdm import tqdm
import numpy as np
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

# 标签列表
LABELS = [
    "car", "sports", "entertainment", "travel", "finance", "story", "game",
    "edu", "culture", "house", "military", "tech", "world", "agriculture", "stock"
]


# 解析数据文件
def load_data(file_path, is_test=False):
    print(f"Loading data from: {file_path}")  # 添加调试输出
    data = []
    with open(file_path, "r", encoding="utf-8") as file:
        for line_num, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                print(f"Warning: Line {line_num} is empty and will be skipped.")
                continue

            parts = line.split("_separator_")
            if len(parts) != 4 and not is_test:
                print(f"Error: Line {line_num} has incorrect parts: {parts}")
                continue

            if is_test:
                data.append({"id": int(parts[0]), "title": parts[1], "key_word": parts[2]})
            else:
                try:
                    label_idx = int(parts[3])
                    if label_idx < 0 or label_idx >= len(LABELS):
                        print(f"Error: Line {line_num} has invalid label index: {label_idx}")
                        continue
                    labels = [1 if i == label_idx else 0 for i in range(len(LABELS))]
                    data.append({
                        "id": int(parts[0]),
                        "title": parts[1],
                        "key_word": parts[2] if parts[2] != "nan" else "无关键字",
                        **dict(zip(LABELS, labels))
                    })
                except ValueError:
                    print(f"Error: Line {line_num} has non-integer label index: {parts[3]}")
                    continue
    return pd.DataFrame(data)


# 数据集加载和预处理
#通过分词器（tokenizer）转化成 token IDs 的数字序列,实现前期对数据进行处理
class MultiLabelDataset(Dataset):
    def __init__(self, data, tokenizer, max_len, is_test=False):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.is_test = is_test

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]
        title = row['title']
        keyword = row['key_word']
        text = f"{title} [SEP] {keyword}"

        inputs = self.tokenizer(
            text,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        if self.is_test:
            return {
                "input_ids": inputs["input_ids"].squeeze(0),
                "attention_mask": inputs["attention_mask"].squeeze(0),
                "id": row["id"]
            }
        else:
            labels = row[LABELS].values.astype(float)
            return {
                "input_ids": inputs["input_ids"].squeeze(0),
                "attention_mask": inputs["attention_mask"].squeeze(0),
                "labels": torch.tensor(labels, dtype=torch.float)
            }




#对于虚假信息检验，分类任务是一个二分类任务
model = BertForSequenceClassification.from_pretrained('bert-base-chinese', num_labels=2)  # num_labels是你的分类任务中的标签数量
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

# 数据集准备部分
train_dataset =pd.read_csv(r"E:\train_and_testA\A\train.txt",encoding='utf-8',sep='_separator_',engine='python')# 这里加载你的训练数据
val_dataset =pd.read_csv(r"E:\train_and_testA\A\test1.txt",encoding='utf-8',sep='_separator_',engine='python')# 这里加载你的验证数据

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)

trainer.train()




# 模型定义
class MultiLabelClassifier:
    def __init__(self, model_name="bert-base-chinese", num_labels=len(LABELS), checkpoint_path="checkpoint.pt"):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
        self.checkpoint_path = checkpoint_path

        # 初始化优化器
        self.optimizer = AdamW(self.model.parameters(), lr=2e-5)

        if os.path.exists(checkpoint_path):
            self.load_checkpoint()

    def load_checkpoint(self):
        checkpoint = torch.load(self.checkpoint_path, map_location='cpu')
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print("Checkpoint loaded successfully.")

    def save_checkpoint(self):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, self.checkpoint_path)
        print("Checkpoint saved successfully.")

    def train(self, train_data, val_data, epochs=1, batch_size=16, learning_rate=2e-5, max_len=128):
        train_dataset = MultiLabelDataset(train_data, self.tokenizer, max_len)
        val_dataset = MultiLabelDataset(val_data, self.tokenizer, max_len)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)

        self.optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        loss_fn = torch.nn.BCEWithLogitsLoss()



        for epoch in range(epochs):
            self.model.train()
            train_loss = 0
            for batch in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}"):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                self.optimizer.zero_grad()
                outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()
            print(f"Epoch {epoch + 1} Training Loss: {train_loss / len(train_loader)}")

            # 验证
            self.evaluate(val_loader, device)

        self.save_checkpoint()

    def evaluate(self, data_loader, device):
        self.model.eval()
        preds = []
        true_labels = []

        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                outputs = self.model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits

                preds.extend(torch.sigmoid(logits).cpu().numpy())
                true_labels.extend(labels.cpu().numpy())

        preds = np.array(preds) > 0.5
        true_labels = np.array(true_labels)

        print("F1 Score:", f1_score(true_labels, preds, average="macro"))
        print("Classification Report:")
        print(classification_report(true_labels, preds, target_names=LABELS))

    def predict(self, test_data, batch_size=16, max_len=128):
        test_dataset = MultiLabelDataset(test_data, self.tokenizer, max_len, is_test=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        self.model.eval()

        results = []
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                ids = batch["id"].numpy()

                outputs = self.model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                probs = torch.sigmoid(logits).cpu().numpy()

                # 获取每个样本的最高概率标签的索引
                max_probs = np.max(probs, axis=1)
                preds = np.argmax(probs, axis=1)

                for i, (pred, max_prob) in zip(ids, zip(preds, max_probs)):
                    if max_prob < 0.5:  # 如果最高概率低于阈值，设为-1
                        results.append({"id": i, "label": -1})
                    else:
                        results.append({"id": i, "label": pred})

        return results


# 主程序
if __name__ == "__main__":
    # 数据文件路径
    #data_files = [f"data_{i}.txt" for i in range(10)]  # 假设有10个数据文件

    # 检查文件是否存在
    #existing_files = [file for file in data_files if os.path.exists(file)]
    #if not existing_files:
        #print("No valid data files found.")
        #exit(1)

    # 读取当前训练状态
    if os.path.exists(r"E:\train_and_testA\A\train.txt"):
        with open(r"E:\train_and_testA\A\train.txt", "r",encoding='utf-8') as f:
            current_file_index =f.read().strip()
            #current_file_index = int(f.read().strip())
    else:
        current_file_index = 0

    # 初始化模型
    model = MultiLabelClassifier()

    # 训练当前文件的数据
    while current_file_index < len(existing_files):
        file_path = existing_files[current_file_index]
        print(f"Training file {file_path}")

        try:
            train_data = load_data(file_path)
            # 训练当前文件
            model.train(train_data, train_data, epochs=1, batch_size=16, learning_rate=2e-5)
        except FileNotFoundError:
            print(f"File not found: {file_path}. Skipping...")
            current_file_index += 1
            continue

        # 更新当前训练状态
        current_file_index += 1
        with open("training_status.txt", "w") as f:
            f.write(str(current_file_index))

    print("All available files have been trained.")

    # 测试
    test_data = load_data("text.txt", is_test=True)
    predictions = model.predict(test_data)
    print("Predictions:")
    for prediction in predictions:
        print(f"Sample {prediction['id']}: {LABELS[prediction['label']] if prediction['label'] != -1 else 'No Label'}")

    # 保存结果为CSV文件
    output_df = pd.DataFrame(predictions)
    output_df.to_csv("drill.csv", index=False, columns=["id", "label"], header=["id", "label"])

    print("Predictions saved to predictions.csv")