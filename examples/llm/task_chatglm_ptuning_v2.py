#! -*- coding: utf-8 -*-
# chatglm的指令微调, 目前评估evaluate有点慢，可以限制只评估部分数据

from bert4torch.models import build_transformer_model
from bert4torch.snippets import sequence_padding, text_segmentate
from bert4torch.callbacks import Callback
import torch.nn as nn
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch
from bert4torch.models import build_transformer_model, BaseModel
from transformers import AutoTokenizer
from bert4torch.snippets import ListDataset, SeqGeneration
import json
import jieba 
from rouge_chinese import Rouge
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import numpy as np
from tqdm import tqdm

# 基本参数
max_source_length = 64
max_target_length = 64
lr = 2e-2
batch_size = 1
grad_accumulation_steps = 16
max_seq_length = max_source_length + max_target_length
ignore_pad_token_for_loss = True
epochs = 4
prefix = ''
prompt_column = 'content'
response_column = 'summary'
history_column = None

# 模型配置
dir_path = "F:/Projects/pretrain_ckpt/chatglm/6B"
config_path = dir_path + '/bert4torch_config.json'
checkpoint_path = [dir_path + f'/bert4torch_pytorch_model_{i}.bin' for i in range(1,9)]  # 可加载单个，也可以加载多个
device = 'cuda' if torch.cuda.is_available() else 'cpu'

tokenizer = AutoTokenizer.from_pretrained(dir_path.replace('/', '\\'), trust_remote_code=True)

# 加载数据集
class MyDataset(ListDataset):
    @staticmethod
    def load_data(filename):
        """加载数据，并尽量分为不超过maxlen的句子
        """
        D = []
        with open(filename, encoding='utf-8') as f:
            for l in f:
                l = json.loads(l)
                prompt, response = l[prompt_column], l[response_column]
                history = l.get('history_column', None)
                D.append((prompt, response, history))
        return D

def build_prompt(query, history):
    if history_column is None:
        prompt = query
    else:
        prompt = ""
        for i, (old_query, answer) in enumerate(history):
            prompt += "[Round {}]\n问：{}\n答：{}\n".format(i, old_query, answer)
        prompt += "[Round {}]\n问：{}\n答：".format(len(history), query)
    return prompt

def collate_fn(batch):
    batch_token_ids, batch_labels = [], []
    for query, answer, history in batch:
        prompt = build_prompt(query, history)
        prompt = prefix + prompt
        a_ids = tokenizer.encode(text=prompt, add_special_tokens=False)
        b_ids = tokenizer.encode(text=answer, add_special_tokens=False)

        if len(a_ids) > max_source_length - 1:
            a_ids = a_ids[:max_source_length - 1]

        if len(b_ids) > max_target_length - 2:
            b_ids = b_ids[:max_target_length - 2]

        input_ids = tokenizer.build_inputs_with_special_tokens(a_ids, b_ids)
        context_length = input_ids.index(tokenizer.bos_token_id)
        mask_position = context_length - 1
        labels = [-100] * context_length + input_ids[mask_position+1:]
        batch_token_ids.append(input_ids)
        batch_labels.append(labels)

    batch_token_ids = torch.tensor(sequence_padding(batch_token_ids, value=tokenizer.pad_token_id), dtype=torch.long, device=device)
    batch_labels = torch.tensor(sequence_padding(batch_labels, value=-100), dtype=torch.long, device=device)
    return [batch_token_ids], batch_labels

train_dataloader = DataLoader(MyDataset('F:/Projects/data/corpus/prompt/AdvertiseGen/train.json'), batch_size=batch_size, shuffle=True, collate_fn=collate_fn) 
dev_dataset = MyDataset('F:/Projects/data/corpus/prompt/AdvertiseGen/dev.json')

class PrefixEncoder(torch.nn.Module):
    """
    The torch.nn model to encode the prefix
    Input shape: (batch-size, prefix-length)
    Output shape: (batch-size, prefix-length, 2*layers*hidden)
    """

    def __init__(self, config):
        super().__init__()
        self.prefix_projection = config.prefix_projection
        if self.prefix_projection:
            # Use a two-layer MLP to encode the prefix
            self.embedding = torch.nn.Embedding(config.pre_seq_len, config.hidden_size)
            self.trans = torch.nn.Sequential(
                torch.nn.Linear(config.hidden_size, config.hidden_size),
                torch.nn.Tanh(),
                torch.nn.Linear(config.hidden_size, config.num_hidden_layers * config.hidden_size * 2)
            )
        else:
            self.embedding = torch.nn.Embedding(config.pre_seq_len, config.num_hidden_layers * config.hidden_size * 2)

    def forward(self, prefix: torch.Tensor):
        if self.prefix_projection:
            prefix_tokens = self.embedding(prefix)
            past_key_values = self.trans(prefix_tokens)
        else:
            past_key_values = self.embedding(prefix)
        return past_key_values
    
class Model(BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.encoder = build_transformer_model(config_path=config_path, checkpoint_path=checkpoint_path, model='glm', token_pad_ids=tokenizer.pad_token_id).half().quantize(4)
        self.config = self.encoder.configs
        self.config.pre_seq_len = 128
        self.config.prefix_projection = False
        for param in self.parameters():
            param.requires_grad = False
        self.prefix_tokens = torch.arange(self.config.pre_seq_len).long()
        self.prefix_encoder = PrefixEncoder(self.config)
        self.dropout = torch.nn.Dropout(0.1)

    def forward(self, token_ids):
        batch_size = token_ids.shape[0]
        prefix_tokens = self.prefix_tokens.unsqueeze(0).expand(batch_size, -1).to(token_ids.device)
        past_key_values = self.prefix_encoder(prefix_tokens).type(torch.float16)
        past_key_values = past_key_values.view(
            batch_size,
            self.config.pre_seq_len,
            self.config.num_hidden_layers * 2,
            self.config.num_attention_heads,
            self.config.hidden_size // self.config.num_attention_heads
        )
        # b, nh, seq_len, hidden_size
        past_key_values = self.dropout(past_key_values)
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)
        past_key_values = [(v[0], v[1]) for v in past_key_values]

        logits = self.encoder([token_ids], past_key_values=past_key_values)
        return logits
model = Model().to(device)
print(f"Number of trainable parameters = {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

class CrossEntropyLoss(nn.CrossEntropyLoss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def forward(self, logits, labels):
        '''
        logits: [btz, seq_len, vocab_size]
        labels: token_ids: [btz, seq_len]
        '''

        logits = logits[:, :-1, :]  # 预测序列，错开一位
        labels = labels[:, 1:]# 目标token_ids
        
        logits = logits.reshape(-1, logits.shape[-1])
        labels = labels.flatten()
        return super().forward(logits, labels)
model.compile(loss=CrossEntropyLoss(ignore_index=-100), optimizer=optim.Adam(model.parameters(), lr), grad_accumulation_steps=grad_accumulation_steps)

class Chat(SeqGeneration):
    def pre_process(self, text):
        return [tokenizer.encode(text)]
    def post_process(self, output_ids):
        return tokenizer.decode(output_ids[0].cpu().numpy())
generation = Chat(model, tokenizer, start_id=None, end_id=tokenizer.encode(['<eop>'])[0], mode='random_sample',
                  maxlen=512, default_rtype='logits', use_states=False)

class Evaluator(Callback):
    """评估与保存
    """
    def __init__(self):
        self.best = 0

    def on_epoch_end(self, steps, epoch, logs=None):
        score_dict = self.evaluate(dev_dataset.data)
        # 保存最优
        if score_dict['bleu-4'] > self.best:
            self.best = score_dict['bleu-4']
            # model.save_weights('./best_model.pt')         
        score_dict['best'] = self.best
        print(score_dict)
    
    def evaluate(self, data):
        preds, labels = [], []
        for query, label, history in tqdm(data, desc='Evaluating'):
            prompt = build_prompt(query, history)
            pred = generation.generate(prompt, topk=50, topp=0.7, temperature=0.95)
            preds.append(pred)
            labels.append(label)

        score_dict = {"rouge-1": [], "rouge-2": [], "rouge-l": [], "bleu-4": []}
        for pred, label in zip(preds, labels):
            hypothesis = list(jieba.cut(pred))
            reference = list(jieba.cut(label))
            rouge = Rouge()
            scores = rouge.get_scores(' '.join(hypothesis) , ' '.join(reference))
            result = scores[0]
            
            for k, v in result.items():
                score_dict[k].append(round(v["f"] * 100, 4))
            bleu_score = sentence_bleu([list(label)], list(pred), smoothing_function=SmoothingFunction().method3)
            score_dict["bleu-4"].append(round(bleu_score * 100, 4))

        for k, v in score_dict.items():
            score_dict[k] = float(np.mean(v))
        return score_dict


if __name__ == '__main__':
    evaluator = Evaluator()

    model.fit(
        train_dataloader,
        steps_per_epoch=None,
        epochs=epochs,
        callbacks=[evaluator]
    )

else:
    model.load_weights('./best_model.pt')
