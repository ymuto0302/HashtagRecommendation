# テキスト（ツイートの集合）を読み込む
import pickle
with open("df_text.pkl", "rb") as f:
    df_text = pickle.load(f)

import sys
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

from transformers import AutoTokenizer, BertModel
MODEL_NAME = 'cl-tohoku/bert-base-japanese-whole-word-masking' # 日本語の事前学習モデル
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
bert_model = BertModel.from_pretrained(MODEL_NAME).to(device)


def text2vec(text):
    encoding = tokenizer(text,
                        max_length=128,
                        padding='max_length',
                        truncation=True)
    # encoding['labels'] = labels
    encoding = {k: torch.tensor(v) for k, v in encoding.items()}

    #print(encoding.keys())
    # print(encoding)
    input_ids = encoding['input_ids'].reshape(1,-1) # バッチ化
    input_ids = input_ids.to(device)
    # print(input_ids)

    bert_output = bert_model(input_ids=input_ids)
    #,
    #                         attention_mask = encoding['attention_mask'],
    #                         token_type_ids = encoding['token_type_ids'])

    cls = bert_output['last_hidden_state'][:,0,:].reshape(-1, 768)[0] # CLS
    # print(cls)

    return cls.detach().cpu().numpy()

text_vectors = []
    
x = df_text['text'].values

# text2vec('東京五輪で野球を観戦する。')
proc_counter = 0
for xx in x:
    # print(xx)
    vec = text2vec(xx)
    text_vectors.append(vec)
    
    proc_counter += 1
    if proc_counter % 1000 == 0:
        print("{}/{}".format(proc_counter, len(x)), file=sys.stderr)

import numpy as np
text_vectors = np.array(text_vectors)

# 後の利用のために保存
np.save('tweet_vector', text_vectors)
