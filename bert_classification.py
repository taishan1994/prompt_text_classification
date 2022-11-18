import json
import random
from collections import Counter
import numpy as np
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader
from transformers import BertForMaskedLM, BertTokenizer, BertForSequenceClassification, BertConfig, AdamW
import torch


def set_seed(seed=123):
    """
    设置随机数种子，保证实验可重现
    :param seed:
    :return:
    """
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_data():
    with open("data/train.json", "r", encoding="utf-8") as fp:
        data = fp.read()
    data = json.loads(data)
    return data


def analyse_data():
    """
    {'其他': 13993, '喜好': 6697, '厌恶': 5978, '悲伤': 5348, '高兴': 4950, '愤怒': 3167}
    """
    data = get_data()
    data = json.loads(data)
    label2id = {
        "其他": 0,
        "喜好": 1,
        "悲伤": 2,
        "厌恶": 3,
        "愤怒": 4,
        "高兴": 5,
    }
    id2label = {v: k for k, v in label2id.items()}
    labels = set()
    labels_count = []
    for d in data:
        text = d[0]
        label = d[1]
        labels.add(label)
        print("".join(text.split(" ")).strip(), id2label[label])
        labels_count.append(id2label[label])
    print(labels)
    counter = Counter(labels_count)
    print(counter)


def load_data():
    data = get_data()
    return_data = []
    # [(文本， 标签id)]
    for d in data:
        text = d[0]
        label = d[1]
        return_data.append(("".join(text.split(" ")).strip(), label))
    return return_data


class Collate:
    def __init__(self,
                 tokenizer,
                 max_seq_len,
                 ):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def collate_fn(self, batch):
        input_ids_all = []
        token_type_ids_all = []
        attention_mask_all = []
        label_all = []
        for data in batch:
            text = data[0]
            label = data[1]
            inputs = self.tokenizer.encode_plus(text=text,
                              max_length=self.max_seq_len,
                              padding="max_length",
                              truncation="longest_first",
                              return_attention_mask=True,
                              return_token_type_ids=True)
            input_ids = inputs["input_ids"]
            token_type_ids = inputs["token_type_ids"]
            attention_mask = inputs["attention_mask"]
            input_ids_all.append(input_ids)
            token_type_ids_all.append(token_type_ids)
            attention_mask_all.append(attention_mask)
            label_all.append(label)

        input_ids_all = torch.tensor(input_ids_all, dtype=torch.long)
        token_type_ids_all = torch.tensor(token_type_ids_all, dtype=torch.long)
        attention_mask_all = torch.tensor(attention_mask_all, dtype=torch.long)
        label_all = torch.tensor(label_all, dtype=torch.long)
        return_data = {
            "input_ids": input_ids_all,
            "attention_mask": attention_mask_all,
            "token_type_ids": token_type_ids_all,
            "label": label_all
        }
        return return_data


class Trainer:
    def __init__(self, args):
        self.args = args
        self.config = BertConfig.from_pretrained(args.model_path, num_labels=6)
        self.model = BertForSequenceClassification.from_pretrained(args.model_path,
                                      config=self.config)
        self.device = args.device
        self.model.to(self.device)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = self.build_optimizer()

    def build_optimizer(self):
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': self.args.weight_decay},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0}
        ]

        # optimizer = AdamW(model.parameters(), lr=learning_rate)
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate)
        return optimizer

    def train(self, train_loader, dev_loader=None):
        self.model.train()
        gloabl_step = 1
        best_acc = 0.
        for epoch in range(1, self.args.epochs + 1):
            for step, batch_data in enumerate(train_loader):
                label = batch_data["label"].to(self.args.device)
                input_ids = batch_data["input_ids"].to(self.args.device)
                token_type_ids = batch_data["token_type_ids"].to(self.args.device)
                attention_mask = batch_data["attention_mask"].to(self.args.device)
                self.optimizer.zero_grad()
                output = self.model(input_ids=input_ids,
                                    token_type_ids=token_type_ids,
                                    attention_mask=attention_mask,
                                    labels=label)
                logits = output[1]
                # 实际上这里计算的loss和output[0]是一样的
                loss = self.criterion(logits, label)
                loss.backward()
                self.optimizer.step()
                print("【train】 epoch：{}/{} step：{}/{} loss：{:.6f}".format(
                    epoch, self.args.epochs, gloabl_step, self.args.total_step, loss.item()
                ))
                gloabl_step += 1
                if gloabl_step % self.args.eval_step == 0:
                    loss, accuracy = self.dev(dev_loader)
                    print("【dev】 loss：{:.6f} accuracy：{:.4f}".format(loss, accuracy))
                    if accuracy > best_acc:
                        best_acc = accuracy
                        print("【best accuracy】 {:.4f}".format(best_acc))
                        torch.save(self.model.state_dict(), "output/bert_classification.pt")
            break

    def dev(self, dev_loader):
        self.model.eval()
        correct_total = 0
        num_total = 0
        loss_total = 0.
        with torch.no_grad():
            for step, batch_data in enumerate(dev_loader):
                label = batch_data["label"].to(self.args.device)
                input_ids = batch_data["input_ids"].to(self.args.device)
                token_type_ids = batch_data["token_type_ids"].to(self.args.device)
                attention_mask = batch_data["attention_mask"].to(self.args.device)
                output = self.model(input_ids=input_ids,
                                    token_type_ids=token_type_ids,
                                    attention_mask=attention_mask,
                                    labels=label)
                logits = output[1]
                loss = self.criterion(logits, label)
                loss_total += loss.item()
                logits = logits.detach().cpu().numpy()
                label = label.view(-1).detach().cpu().numpy()
                num_total += len(label)
                preds = np.argmax(logits, axis=1).flatten()
                correct_num = (preds == label).sum()
                correct_total += correct_num

        return loss_total, correct_total / num_total

    def test(self, model, test_loader, labels):
        model.eval()
        preds = []
        trues = []
        with torch.no_grad():
            for step, batch_data in enumerate(test_loader):
                label = batch_data["label"].to(self.args.device)
                input_ids = batch_data["input_ids"].to(self.args.device)
                token_type_ids = batch_data["token_type_ids"].to(self.args.device)
                attention_mask = batch_data["attention_mask"].to(self.args.device)
                output = model(input_ids=input_ids,
                                    token_type_ids=token_type_ids,
                                    attention_mask=attention_mask,
                                    labels=label)
                logits = output[1].detach().cpu().numpy()
                label = label.view(-1).detach().cpu().numpy().tolist()
                pred = np.argmax(logits, axis=1).flatten().tolist()
                trues.extend(label)
                preds.extend(pred)
        print(trues, preds, labels)
        report = classification_report(trues, preds, target_names=labels)
        return report


class Args:
    model_path = "model_hub/chinese-bert-wwm-ext"
    max_seq_len = 128
    ratio = 0.8
    device = torch.device("cuda" if torch.cuda.is_available else "cpu")
    train_batch_size = 32
    dev_batch_size = 32
    weight_decay = 0.01
    epochs = 1
    learning_rate = 3e-5
    eval_step = 100


def main():
    set_seed()
    args = Args()
    tokenizer = BertTokenizer.from_pretrained(args.model_path)
    data = load_data()
    random.shuffle(data)
    train_num = int(len(data) * args.ratio)
    train_data = data[:train_num]
    dev_data = data[train_num:]


    label2id = {
        "其他": 0,
        "喜好": 1,
        "悲伤": 2,
        "厌恶": 3,
        "愤怒": 4,
        "高兴": 5,
    }

    collate = Collate(tokenizer, args.max_seq_len)
    train_loader = DataLoader(train_data,
                              batch_size=args.train_batch_size,
                              shuffle=True,
                              num_workers=2,
                              collate_fn=collate.collate_fn)
    total_step = len(train_loader) * args.epochs
    args.total_step = total_step
    dev_loader = DataLoader(dev_data,
                            batch_size=args.dev_batch_size,
                            shuffle=False,
                            num_workers=2,
                            collate_fn=collate.collate_fn)
    test_loader = dev_loader

    trainer = Trainer(args)

    # trainer.train(train_loader, dev_loader)

    labels = list(label2id.keys())
    ckpt_path = "output/bert_classification.pt"
    config = BertConfig.from_pretrained(args.model_path, num_labels=6)
    model = BertForSequenceClassification.from_pretrained(args.model_path, config=config)
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    model.to(args.device)
    report = trainer.test(model, test_loader, labels)
    print(report)

    """
              precision    recall  f1-score   support

          其他       0.62      0.77      0.69      2875
          喜好       0.63      0.62      0.63      1330
          悲伤       0.63      0.48      0.54      1079
          厌恶       0.53      0.34      0.41      1147
          愤怒       0.45      0.53      0.49      649
          高兴       0.66      0.61      0.63      947

    accuracy                           0.61      8027
    macro avg          0.59      0.56      0.57      8027
    weighted avg        0.60      0.61      0.60      8027
    """


if __name__ == '__main__':
    main()
