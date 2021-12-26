# coding=utf-8
import json
import requests


def main():
    # load vocab
    key2id = ...
    val2id = ...
    word2id = ...
    id2word = ...

    input_texts = ""
    keys = []
    vals = []
    key_input_ids = []

    data = {"instances": [
        {
            "key_input_ids": tmp,
            "val_input_ids": tmp,
            "output_ids": tmp
        }
    ]
    }

    param = json.dumps(data)
    # res = requests.post('http://localhost:8502/v1/models/half_plus_two:predict', data=param)
    json_response = requests.post('http://192.168.140.158:8501/v1/models/QA:predict', data=param)
    # text is a DICT
    print(json_response.text)


tmp = [1, 442, 5562, 2, 545,
       1, 442, 5562, 2, 545,
       1, 442, 5562, 2, 545,
       1, 442, 5562, 2, 545,
       1, 442, 5562, 2, 545,
       1, 442, 5562, 2, 545,
       1, 442, 5562, 2, 545,
       1, 442, 5562, 2, 545,
       1, 442, 5562, 2, 545,
       1, 442, 5562, 2, 545]

input_ids = []

for _ in range(16):
    input_ids.append(tmp)

output_ids = input_ids


# data = {"instances": [5, 7, 9]
# }
