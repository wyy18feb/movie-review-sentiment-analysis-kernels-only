from transformers import BertForSequenceClassification, AdamW
from utils.config import bert_config_from_file


config = bert_config_from_file('conf/bert.json')
model = BertForSequenceClassification.from_pretrained(config['pretrained'], cache_dir=config['cache_dir'], num_labels=5)
no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]
optimizer = AdamW(optimizer_grouped_parameters, lr=2e-5, eps=1e-8)
