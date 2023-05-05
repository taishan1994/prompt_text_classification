import json
import random
from collections import Counter
import numpy as np
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader
from transformers import BertForMaskedLM, BertTokenizer, BertForSequenceClassification, BertConfig, AdamW
from transformers import pipeline
import torch
from pprint import pprint

label2id = {
    "其他": 0,
    "喜好": 1,
    "悲伤": 2,
    "厌恶": 3,
    "愤怒": 4,
    "高兴": 5,
}

id2label = {v: k for k, v in label2id.items()}

label2ind = {
    "其他": 1,
    "喜好": 2,
    "悲伤": 3,
    "厌恶": 4,
    "愤怒": 5,
    "高兴": 6,
}


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


def load_data(prompt, max_seq_len):
    data = get_data()
    return_data = []
    # [(文本， 标签id)]
    for d in data:
        text = d[0]
        label = d[1]
        flag = False
        for lab, lab_id in label2id.items():
          tmp_text = "".join(text.split(" ")).strip() + prompt.format(lab)
          if len(text) > max_seq_len - 2:
            continue
          # 0表示不是该类，1表示是该类
          if lab_id == label:
            return_data.append((tmp_text, 1))
          else:
            return_data.append((tmp_text, 0))
    return return_data


class Collate:
    def __init__(self,
                 tokenizer,
                 max_seq_len):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def collate_fn(self, batch):
        input_ids_all = []
        token_type_ids_all = []
        attention_mask_all = []
        label_all = []
        mask_pos_all = []
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
            mask_pos = [i for i, token_id in enumerate(input_ids) if token_id ==
                        self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)]
            mask_pos_all.append(mask_pos)
            token_type_ids = inputs["token_type_ids"]
            attention_mask = inputs["attention_mask"]
            input_ids_all.append(input_ids)
            token_type_ids_all.append(token_type_ids)
            attention_mask_all.append(attention_mask)
            label_all.append(label)

        input_ids_all = torch.tensor(input_ids_all, dtype=torch.long)
        token_type_ids_all = torch.tensor(token_type_ids_all, dtype=torch.long)
        attention_mask_all = torch.tensor(attention_mask_all, dtype=torch.long)
        mask_pos_all = torch.tensor(mask_pos_all, dtype=torch.long)
        label_all = torch.tensor(label_all, dtype=torch.long)
        return_data = {
            "input_ids": input_ids_all,
            "attention_mask": attention_mask_all,
            "token_type_ids": token_type_ids_all,
            "label": label_all,
            "mask_pos": mask_pos_all,
        }
        return return_data


class Trainer:
    def __init__(self, args):
        self.args = args
        self.model = BertForMaskedLM.from_pretrained(args.model_path)
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
                label = batch_data["label"].to(self.device)
                batch_size = label.size(0)
                input_ids = batch_data["input_ids"].to(self.device)
                mask_pos = batch_data["mask_pos"].to(self.device)
                token_type_ids = batch_data["token_type_ids"].to(self.device)
                attention_mask = batch_data["attention_mask"].to(self.device)
                self.optimizer.zero_grad()
                output = self.model(input_ids=input_ids,
                                    token_type_ids=token_type_ids,
                                    attention_mask=attention_mask)
                logits = output.logits
                loss = None

                for i in range(batch_size):
                    mask_pos_tmp = mask_pos[i]
                    mask = mask_pos_tmp[0]
                    pred = logits[i, mask][[7,8]]
                    loss_tmp = self.criterion(pred.unsqueeze(0), label[i].unsqueeze(0))
                    if loss is None:
                      loss = loss_tmp
                    else:
                      loss += loss_tmp
                loss.backward()
                self.optimizer.step()
                print("【train】 epoch：{}/{} step：{}/{} loss：{:.6f}".format(
                    epoch, self.args.epochs, gloabl_step, self.args.total_step, loss.item()
                ))
                gloabl_step += 1
                # if gloabl_step % self.args.eval_step == 0:
                #     loss, accuracy = self.dev(dev_loader)
                #     print("【dev】 loss：{:.6f} accuracy：{:.4f}".format(loss, accuracy))
                #     if accuracy > best_acc:
                #         best_acc = accuracy
                #         print("【best accuracy】 {:.4f}".format(best_acc))
                #         torch.save(self.model.state_dict(), "output/bert_prompt.pt")
        torch.save(self.model.state_dict(), "output/bert_prompt.pt")


    def dev(self, dev_loader):
        self.model.eval()
        correct_total = 0
        num_total = 0
        loss_total = 0.
        with torch.no_grad():
            for step, batch_data in enumerate(dev_loader):
                label = batch_data["label"].to(self.device)
                input_ids = batch_data["input_ids"].to(self.device)
                mask_pos = batch_data["mask_pos"].to(self.device)
                token_type_ids = batch_data["token_type_ids"].to(self.device)
                attention_mask = batch_data["attention_mask"].to(self.device)
                output = self.model(input_ids=input_ids,
                                    token_type_ids=token_type_ids,
                                    attention_mask=attention_mask)
                logits = output.logits
                loss = None
                batch_size = label.size(0)
                correct_num = 0
                for i in range(batch_size):
                    mask_pos_tmp = mask_pos[i]
                    mask = mask_pos_tmp[0]
                    pred = logits[i, mask][[7,8]]
                    logit = pred.detach().cpu().numpy()
                    logit = np.argmax(logit, axis=-1)

                    if logit == label[i].detach().cpu().numpy():
                        correct_num += 1

                    loss_tmp = self.criterion(pred.unsqueeze(0), label[i].unsqueeze(0))
                    if loss is None:
                        loss = loss_tmp
                    else:
                        loss += loss_tmp
                loss_total += loss.item()
                num_total += len(label)
                correct_total += correct_num

        return loss_total, correct_total / num_total

    def test(self, model, test_loader, labels):
        model.eval()
        preds = []
        trues = []
        with torch.no_grad():
            for step, batch_data in enumerate(test_loader):
                label = batch_data["label"].to(self.device)
                mask_pos = batch_data["mask_pos"].to(self.device)
                input_ids = batch_data["input_ids"].to(self.device)
                token_type_ids = batch_data["token_type_ids"].to(self.device)
                attention_mask = batch_data["attention_mask"].to(self.device)
                output = model(input_ids=input_ids,
                               token_type_ids=token_type_ids,
                               attention_mask=attention_mask,
                               )
                logits = output.logits
                batch_size = label.size(0)
                pred_tmp = []
                for i in range(batch_size):
                    mask_pos_tmp = mask_pos[i]
                    mask = mask_pos_tmp[0]
                    pred = logits[i, mask][[7,8]]

                    logit = pred.detach().cpu().numpy()
                    logit = np.argmax(logit, axis=-1)

                    if logit == label[i].detach().cpu().numpy():
                        pred_tmp.append(logit)
                    else:
                        pred_tmp.append(0)
                label = label.detach().cpu().numpy()
                trues.extend(label)
                preds.extend(pred_tmp)
        report = classification_report(trues, preds, target_names=labels)
        """
              precision    recall  f1-score   support

          不是       0.90      1.00      0.95     39799
           是       1.00      0.45      0.62      7803

    accuracy                           0.91     47602
    macro avg       0.95      0.73      0.79     47602
    weighted avg       0.92      0.91      0.90     47602
        """
        return report

def predict():
    args = Args()
    tokenizer = BertTokenizer.from_pretrained(args.model_path)
    ckpt_path = "output/bert_prompt.pt"
    model = BertForMaskedLM.from_pretrained(args.model_path)
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    model.to(args.device)
    masker = pipeline("fill-mask", model=model, tokenizer=tokenizer)
    prompt = "[MASK]{}。"
    texts = [
      ["银色的罗马高跟鞋，圆球吊饰耳饰单带，个性十足，有非常抢眼！", 1],
      ["稳吾到嘛？", 3],
      ["以后打死也不吃了", 3],
      ["来广州两天都没能织围脖,一直都在忙,再加上又感冒了,好痛苦[泪]不过广州给我的感觉灰常好!", 5],
      ["对骂我从来没怕过，你们也就只能考虑暗杀了，这样就充分保护动物了，臭傻逼们[打哈气]", 4],
      ["你这么说的我都不好意思呢", 0],
      ["我到了，文，好惨啊…", 2],
      ["真是不容易啊。。", 0]
    ]
    text_process = []
    for text in texts:
      for lab, lab_id in label2ind.items():
        if lab_id == text[1]:
          text_process.append((text[0].strip()+prompt.format(lab), 1))
        else:
          text_process.append((text[0].strip()+prompt.format(lab), 0))
    collate = Collate(tokenizer, args.max_seq_len)
    batch_data = collate.collate_fn(text_process)
    label = batch_data["label"].to(args.device)
    mask_pos = batch_data["mask_pos"].to(args.device)
    input_ids = batch_data["input_ids"].to(args.device)
    token_type_ids = batch_data["token_type_ids"].to(args.device)
    attention_mask = batch_data["attention_mask"].to(args.device)
    output = model(input_ids=input_ids,
                    token_type_ids=token_type_ids,
                    attention_mask=attention_mask,
                    )
    logits = output.logits
    batch_size = label.size(0)
    pred_tmp = []

    for i in range(batch_size):
        mask_pos_tmp = mask_pos[i]
        mask = mask_pos_tmp[0]
        pred = logits[i, mask][[7,8]]
        sep_pos = input_ids[i].detach().cpu().numpy().tolist().index(102)
        pred_label = tokenizer.decode(input_ids[i][mask+1:sep_pos])
        logit = pred.detach().cpu().numpy()
        logit_score = torch.nn.functional.softmax(pred, dim=-1)
        logit_ind = np.argmax(logit, axis=-1)

        label_tmp = (label[i].detach().cpu().numpy().tolist())
        if logit_ind == 1:
          print(tokenizer.decode(input_ids[i][:sep_pos]))
          print("预测：", logit_ind, logit_score[logit_ind].item(), pred_label)
          print("=" * 100)

class Args:
    model_path = "./model_hub/chinese-bert-wwm-ext"
    max_seq_len = 128
    ratio = 0.8
    device = torch.device("cuda" if torch.cuda.is_available else "cpu")
    train_batch_size = 32
    dev_batch_size = 32
    weight_decay = 0.01
    epochs = 1
    learning_rate = 3e-5
    eval_step = 500
    prompt = "[MASK]{}。"


def main():
    set_seed()
    args = Args()
    tokenizer = BertTokenizer.from_pretrained(args.model_path)
    data = load_data(prompt=args.prompt, max_seq_len=args.max_seq_len)
    random.shuffle(data)
    train_num = int(len(data) * args.ratio)
    train_data = data[:train_num]
    dev_data = data[train_num:]

    label2id = {
        "不是": 0,
        "是": 1,
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
                            shuffle=True,
                            num_workers=2,
                            collate_fn=collate.collate_fn)
    test_loader = dev_loader

    trainer = Trainer(args)

    trainer.train(train_loader, dev_loader)

    labels = label2id.keys()
    ckpt_path = "output/bert_prompt.pt"
    model = BertForMaskedLM.from_pretrained(args.model_path)
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    model.to(args.device)
    report = trainer.test(model, test_loader, labels)
    print(report)

def test_load_data():
    data = load_data(prompt = "[MASK]{}。", max_seq_len=128)
    print(data[0])

if __name__ == '__main__':
    # main()
    predict()
    # test_load_data()