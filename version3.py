import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
from sklearn.metrics import f1_score, classification_report
import pandas as pd
from tqdm import tqdm
import numpy as np
import os
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR



# 标签列表
LABELS = [
    "car", "sports", "entertainment", "travel", "finance", "story", "game",
    "edu", "culture", "house", "military", "tech", "world", "agriculture", "stock"
]

# 解析数据文件
#将数据文件解析为可被用的文件类型
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

#早停法
class EarlyStopping:
    def __init__(self, patience=5, min=0, path='checkpoint.pt'):
        """
        :param patience: 当验证集性能不再提升时，等待多少个 epoch 后停止训练
        :param min: 验证集性能改善的最小阈值（如果验证损失降低少于该值，则认为没有提升）
        :param path: 用于保存模型的路径（当性能最好时保存模型）
        """
        self.patience = patience  # 最大等待的 epoch 数
        self.min = min  # 性能改善的最小阈值
        self.path = path  # 模型检查点保存路径
        self.best_score = None  # 最好的验证损失（初始化为 None）
        self.early_stop_count = 0  # 连续没有改善的 epoch 数
        self.best_model_state = None  # 最佳模型的状态字典

    def __call__(self, val_loss, model):
        """
        :param val_loss: 当前验证集的损失
        :param model: 当前的模型
        :return: 是否触发早停
        """
        if self.best_score is None:
            # 如果没有最佳性能，初始化为当前验证损失
            self.best_score = val_loss
            self.save_checkpoint(model)
        elif val_loss < self.best_score - self.min:
            # 如果当前验证损失比最佳性能好，并且有明显改进（大于 min）
            self.best_score = val_loss
            self.early_stop_count = 0  # 重置计数器
            self.save_checkpoint(model)  # 保存最佳模型
        else:
            # 否则，增加早停计数器
            self.early_stop_count += 1
            if self.early_stop_count >= self.patience:
                # 如果连续 patience 个 epoch 没有改善，触发早停
                print(f'Early stopping triggered after {self.patience} epochs without improvement.')
                return True  # 返回 True 表示训练应该停止
        return False  # 否则，返回 False，表示继续训练

    def save_checkpoint(self, model):
        """保存当前模型的权重"""
        self.best_model_state = model.state_dict()  # 获取当前模型的状态字典
        torch.save(self.best_model_state, self.path)  # 保存为 checkpoint 文件
        print(f'Model checkpoint saved at {self.path}')
# 模型定义
#定义优化器和学习率调度器
class MultiLabelClassifier(nn.Module):
    def __init__(self, model_name="bert-base-chinese", num_labels=len(LABELS), checkpoint_path="checkpoint.pt"):
        super(MultiLabelClassifier, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
        self.checkpoint_path = checkpoint_path
        self.epoch = 0
        self.loss = torch.nn.BCEWithLogitsLoss()
        self.optimizer = AdamW(self.model.parameters(), lr=2e-5)
        if os.path.exists(checkpoint_path):
            self.load_checkpoint()

    #加载检查点
    def load_checkpoint(self):
        checkpoint = torch.load(self.checkpoint_path, map_location='cpu')
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epoch = checkpoint['epoch']
        print("Checkpoint loaded successfully.")


    def save_checkpoint(self):
        checkpoint = {
            'epoch': self.epoch,  # 当前训练轮数
            'model_state_dict': self.model.state_dict(),  # 模型的状态字典
            'optimizer_state_dict': self.optimizer.state_dict(),  # 优化器的状态字典
            'loss': self.loss  # 当前的训练损失（可以根据需要保存其他信息）
        }

        # 保存字典为文件
        torch.save(checkpoint, self.checkpoint_path)
        print(f"Checkpoint saved at {self.checkpoint_path}")

        print("Checkpoint saved successfully.")


     # 根据需要设置训练轮数
    def train(self, train_data, val_data, epochs=3, batch_size=32, learning_rate=2e-5, max_len=128,patience=3,step_size=2,gamma=0.1):
        train_dataset = MultiLabelDataset(train_data, self.tokenizer, max_len)
        val_dataset = MultiLabelDataset(val_data, self.tokenizer, max_len)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)

        self.optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        scheduler = StepLR(self.optimizer, step_size=step_size, gamma=gamma)

        early_stopping=EarlyStopping(patience=patience,path=self.checkpoint_path)
        loss_fn = torch.nn.BCEWithLogitsLoss()



        for epoch in range(epochs):
            self.model.train()#设置模型为训练模式
            train_loss = 0
            for batch in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}"):
                self.optimizer.zero_grad()#清空阶梯
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                #模型向前传播
                outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                #反向传播
                loss.backward()
                #更新参数
                self.optimizer.step()

                train_loss += loss.item()
            print(f"Epoch {epoch + 1} Training Loss: {train_loss / len(train_loader)}")

            # 验证
            val_loss=self.evaluate(val_loader, device)
            if early_stopping(val_loss,self.model):
                print(f"Stopping training early at epoch {epoch + 1}.")
                break
            #保存模型检查点
            self.save_checkpoint()
            #更新学习率
            scheduler.step()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    def evaluate(self, data_loader, device):
        self.model.eval()
        preds = []
        true_labels = []
        total_loss=0

        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                outputs = self.model(input_ids, attention_mask=attention_mask,labels=labels)
                logits = outputs.logits
                loss=outputs.loss
                total_loss+=loss.item()

                preds.extend(torch.sigmoid(logits).cpu().numpy())
                true_labels.extend(labels.cpu().numpy())

        avg_val_loss = total_loss / len(data_loader)
        print(f"Validation Loss: {avg_val_loss}")

        preds = np.array(preds) > 0.5
        true_labels = np.array(true_labels)


        print("F1 Score:", f1_score(true_labels, preds, average="macro"))
        print("Classification Report:")
        print(classification_report(true_labels, preds, target_names=LABELS))

    def predict(self, test_data, batch_size=32, max_len=128):
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
    #data_file=r"D:\train.txt"
    #train_data=load_data(data_file)
    #val_file=r"D:\python_study\pythonProject\output\train_part_1.txt"
    #val_data=load_data(val_file)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 初始化模型
    model = MultiLabelClassifier(model_name="bert-base-chinese", num_labels=len(LABELS), checkpoint_path="checkpoint.pt")
    #model.train(train_data, val_data, epochs=5, step_size=3, gamma=0.1)



    # 测试
    test_data = load_data(r"D:\REE\B\test2.txt", is_test=True)
    predictions = model.predict(test_data)
    print("Predictions:")
    for prediction in predictions:
        print(f"Sample {prediction['id']}: {LABELS[prediction['label']] if prediction['label'] != -1 else 'No Label'}")
    # 保存结果为CSV文件
    output_df = pd.DataFrame(predictions)
    output_df.to_csv("drill.csv", index=False, columns=["id", "label"], header=["id", "label"])

    print("Predictions saved to predictions.csv")




