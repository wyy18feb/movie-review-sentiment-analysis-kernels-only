from transformers import BertTokenizer
from utils.config import bert_config_from_file

config = bert_config_from_file("conf/bert.json")
tokenizer = BertTokenizer.from_pretrained(config['pretrained'], cache_dir=config['cache_dir'], do_lower_case=True)
