import glob
import json
import os.path
import jieba
import Levenshtein
import numpy as np
from string import punctuation
import argparse

import spacy
from text2vec import SentenceModel
from sentence_transformers import SentenceTransformer, util
import nltk
import benepar
benepar.download('benepar_zh2')
import pandas as pd
from tqdm import tqdm
from utils import *


def spacy_process(text):
    return SpacyModel(str(text))


def get_dependency_tree_depth(sentence):
    def get_subtree_depth(node):
        if len(list(node.children)) == 0:
            return 0
        return 1 + max([get_subtree_depth(child) for child in node.children])

    tree_depths = [get_subtree_depth(spacy_sentence.root) for spacy_sentence in spacy_process(sentence).sents]
    if len(tree_depths) == 0:
        return 0
    return max(tree_depths)


def remove_punctuation_characters(text):
    return ''.join([char for char in text if char not in punctuation])


def is_punctuation(word):
    return remove_punctuation_characters(word) == ''


def get_levenshtein_similarity(complex_sentence, simple_sentence):
    return Levenshtein.ratio(complex_sentence, simple_sentence)


def to_words(sentence):
    return sentence.split()


def remove_stopwords(text):
    return ' '.join([w for w in to_words(text) if w.lower() not in stopwords])


def remove_punctuation_tokens(text):
    return ' '.join([w for w in to_words(text) if not is_punctuation(w)])


def get_rank(word):
    return Word2Rank.get(word, len(Word2Rank))


def get_log_rank(word):
    return np.log(1 + get_rank(word))


def get_lexical_complexity_score(sentence):
    words = to_words(remove_stopwords(remove_punctuation_tokens(sentence)))
    words = [word for word in words if word in Word2Rank]
    if len(words) == 0:
        return np.log(1 + len(Word2Rank))  # TODO: This is completely arbitrary
    return np.quantile([get_log_rank(word) for word in words], 0.75)


def min_max_scaler(dataList):
    maxNum = max(dataList)
    minNum = min(dataList)
    return [(i-minNum)/(maxNum-minNum) for i in dataList]


def z_score_scaler(dataList):
    dataList = np.array(dataList)
    return [float((x - dataList.mean())/dataList.std()) for x in dataList]


def get_tree_string(doc):
    return next(iter(doc.sents))._.parse_string


def get_syn_div(predictions, references, batch_size = 1):
    with SpacyModel.select_pipes(enable = ['parser', 'benepar']):
        preds = list(tqdm(SpacyModel.pipe(predictions, batch_size = batch_size), total = len(predictions),
                          desc = "syntdiv:parse_preds", disable = True))
        preds = list(map(get_tree_string, preds))
        refs = list(tqdm(SpacyModel.pipe(references, batch_size = batch_size), total = len(references),
                         desc = "syntdiv:parse_refs", disable = True))
        refs = list(map(get_tree_string, refs))

    scores = list(tqdm(map(dist, zip(preds, refs)), total = len(preds), desc = "syntdiv:calc_dist"))
    return scores[0]


if __name__ == '__main__':
    SemModel = SentenceModel("C:/PLMs/text2vec-base-chinese")
    df = pd.read_csv("data/pku_features_50w_add-distilbert-multilingual.csv")
    paraphrase1, paraphrase2 = [], []
    paraphrase1_cws, paraphrase2_cws = [], []
    lexsim, semsim = [], []
    for rid, row in df.iterrows():
        para1, para2 = row["paraphrase1"], row["paraphrase2"]
        paraphrase1.append(para1)
        paraphrase2.append(para1)
        para1_cws = ' '.join(jieba.lcut(para1))
        para2_cws = ' '.join(jieba.lcut(para2))
        paraphrase1_cws.append(para1_cws)
        paraphrase2_cws.append(para2_cws)
        lex = get_levenshtein_similarity(para1, para2)
        lexsim.append(lex)
        embeddings = SemModel.encode([para1, para2], convert_to_tensor = True)
        sim = float(util.pytorch_cos_sim(embeddings[0], embeddings[1]))
        semsim.append(sim)
    synsim = 1 - get_syn_div([paraphrase1_cws], [paraphrase2_cws])  # 句法相似度
    para_save = pd.DataFrame({
        "paraphrse1": paraphrase1, "paraphrase2": paraphrase2,
        "levsim": lexsim, "synsim": synsim, "semsim": semsim
    })
    para_save.to_csv("data/para3sim.csv", index = False, sep = ',')



