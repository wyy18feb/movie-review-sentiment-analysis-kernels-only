import os
import json


def _read_json_from_file(filename):
    assert type(filename) is str
    exists = os.path.isfile(filename)
    assert exists
    with open(filename, 'r') as f:
        conf_str = f.read()
    json_config = json.loads(conf_str)
    return json_config


def bert_config_from_file(filename):
    json_config = _read_json_from_file(filename)
    assert json_config.get('pretrained') is not None
    assert json_config.get('cache_dir') is not None
    return json_config


def dataset_config_from_file(filename):
    json_config = _read_json_from_file(filename)
    assert json_config.get('path') is not None and type(json_config['path']) is str and os.path.isfile(json_config['path'])
    json_config['limit'] = json_config.get('limit')
    assert type(json_config['limit']) in [int, type(None)]
    assert json_config.get('batch_size') is not None
    train_size = json_config['batch_size'].get('train')
    assert train_size is not None and type(train_size) is int
    evaluate_size = json_config['batch_size'].get('evaluate')
    assert evaluate_size is not None and type(evaluate_size) is int
    test_size = json_config['batch_size'].get('test')
    assert test_size is not None and type(test_size) is int
    return json_config
