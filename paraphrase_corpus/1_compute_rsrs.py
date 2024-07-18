import glob
import json
import os.path
import argparse

import pandas as pd
from transformers import BertForMaskedLM, AutoTokenizer, AutoModelForMaskedLM
import torch
import torch.nn.functional as F
import math
from tqdm import tqdm


def compute_wnll(yt, yp):
    wnll = -((yt.dot(torch.log(yp))) + ((1-yt).dot(torch.log(1-yp))))
    # try: wnll = -((yt*torch.log(yp)) + ((1-yt)*torch.log(1-yp)))
    # except: wnll = -((yt*torch.log(yp+1e-10)) + ((1-yt)*torch.log(1-yp+1e-10)))
    return float(wnll)


def compute_rsrs(wnll):
    wnll.sort()
    rsrs = 0
    for i, w in enumerate(wnll):
        squ_root = math.sqrt(i + 1)
        rsrs += squ_root*w
    return rsrs/len(wnll)


def compute_readability(sentence):
    WNLL = []
    for id, token in enumerate(sentence):
        sentence_mask = sentence[:id] + '[MASK]' + sentence[id + 1:]
        inputs = tokenizer(sentence_mask, return_tensors = 'pt').to(args.device)
        for idx, i in enumerate(inputs.input_ids[0]):
            if int(i) == 103:
                mask_id = idx
                break
        output = model(**inputs)[0]
        # output_id = torch.argmax(output, -1)
        # pred_seq = tokenizer.convert_ids_to_tokens(output_id[0])
        # pred_token = pred_seq[mask_id]
        orig_token_id = tokenizer(token).input_ids[1]
        yp = F.softmax(output[0, mask_id, :], dim = 0)
        yt = torch.zeros(len(tokenizer)).to(args.device)
        yt[orig_token_id] = 1
        wnll = compute_wnll(yt, yp)
        WNLL.append(wnll)
    assert len(WNLL) == len(sentence)
    return compute_rsrs(WNLL)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default = 'your/path/to/distilbert-base-multilingual-cased')
    parser.add_argument('--device', default = 'cuda')
    args = parser.parse_args()
    model = AutoModelForMaskedLM.from_pretrained(args.model_path).to(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model.eval()
    complex_rdb = []
    simple_rdb = []
    corpus = pd.read_csv("data/pku_features_50w_add-distilbert-multilingual.csv")
    source = []
    target = []
    rdb1_list = []
    rdb2_list = []
    for rid, line in tqdm(corpus.iterrows()):
        para1 = line['paraphrase1']
        para2 = line['paraphrase2']
        para1_rsrs = compute_readability(para1)
        para2_rsrs = compute_readability(para2)
        rdb1_list.append(para1_rsrs)
        rdb2_list.append(para2_rsrs)
    corpus['rsrs1 disitl'] = rdb1_list
    corpus['rsrs2 distil'] = rdb2_list
    corpus.to_csv("data/para3sim.csv", index = False, sep = ',')



