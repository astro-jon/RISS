import csv
import json
import pickle
from itertools import groupby

import pandas as pd
from apted import APTED
from apted.helpers import Tree
import numpy as np
from tqdm import tqdm


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def load_pickle(datadir):
    file = open(datadir, 'rb')
    data = pickle.load(file)
    return data


def data_analyse(diff, section):
    max_diff = max(diff)
    min_diff = min(diff)
    diff.sort()  # 自身排序
    split_section = {}
    check_len = 0
    for k, g in groupby(diff, key = lambda x: x // section):
        start = '{:.2f}'.format(k*section)
        end = '{:.2f}'.format((k+1)*section)
        str_section = start + '~' + end
        list_g = list(g)
        check_len += len(list_g)  # 一次性使用变量
        split_section[str_section] = list_g
    assert check_len == len(diff)
    return max_diff, min_diff, split_section


def get_section_list(section, pred_list, section_list):
    return_list = []
    for i in section_list:
        if i not in section:
            return_list.append(0.0)
        else:
            return_list.append(float('{:.5f}'.format(len(section[i]) / len(pred_list))))
    return return_list


def min_max_scaler(dataList):
    maxNum = max(dataList)
    minNum = min(dataList)
    return [(i-minNum)/(maxNum-minNum) for i in dataList]


def data_normalize(dataList, standard = False):
    if standard:
        dataList = np.array(dataList)
        dataList = [float((x - dataList.mean())/dataList.std()) for x in dataList]
    return min_max_scaler(dataList)


def dist(pair):
    p_tree_n = normalize_tree(pair[0], max_depth = 3)
    r_tree_n = normalize_tree(pair[1], max_depth = 3)

    ted = tree_edit_distance(p_tree_n, r_tree_n)

    return ted


def normalize_tree(tree_string, max_depth = 3):
    res = []
    depth = -1
    leaf = False
    for c in tree_string:
        if c in ['{', '}']:
            continue
        if c == '(':
            leaf = False
            depth += 1

        elif c == ')':
            leaf = False
            depth -= 1
            if depth < max_depth:
                res.append('}')
                continue

        elif c == ' ':
            leaf = True
            continue

        if depth <= max_depth and not leaf and c != ')':
            res.append(c if c != '(' else '{')

    return ''.join(res)


def tree_edit_distance(lintree1, lintree2):
    tree1 = Tree.from_text(lintree1)
    tree2 = Tree.from_text(lintree2)
    n_nodes_t1 = lintree1.count('{')
    n_nodes_t2 = lintree2.count('{')

    apted = APTED(tree1, tree2)
    ted = apted.compute_edit_distance()
    return ted / (n_nodes_t1 + n_nodes_t2)


def data_analysis():
    reader = open('paraphrase_unorder/paraphrase_unorder', 'r', encoding = 'utf-8').readlines()
    len_list_1 = []
    len_list_2 = []
    len_over_120_1 = []
    len_over_120_2 = []
    for line in tqdm(reader):
        p1, p2 = line.strip().split('\t')
        len_list_1.append(len(p1))
        len_list_2.append(len(p2))
        if len(p1) > 120: len_over_120_1.append(len(p1))
        if len(p2) > 120: len_over_120_2.append(len(p2))
    print(max(len_list_1), max(len_list_2))


def merge_data():
    data_df = pd.read_csv('data_df.csv')
    data_df2 = pd.read_csv('data_df2.csv')
    syndiv_dict = json.load(open('syndiv.json', 'r', encoding = 'utf-8'))
    row_len = len(syndiv_dict['para1'])
    header = ['paraphrase1', 'paraphrase2', 'levdiff1', 'levdiff2', 'syndiff1', 'syndiff2',
              'synsim', 'nbchars1', 'nbchars2', 'levsim']
    csv_writer = csv.DictWriter(open('PKU-10w-UnerLen120.csv', 'w', encoding = 'utf-8', newline = ''), fieldnames = header)
    csv_writer.writeheader()
    for row1, row2 in zip(data_df.iterrows(), data_df2.iterrows()):
        rid, row1 = row1
        _, row2 = row2
        if rid >= 100000: break
        if len(row1['source']) < 126 and len(row1['reference']) < 126:
            csv_writer.writerow({
                "paraphrase1": row1["source"],
                "paraphrase2": row1['reference'],
                "levdiff1": row1['word2rank1'],
                'levdiff2': row1['word2rank2'],
                "syndiff1": row1["deptreedepth1"],
                "syndiff2": row1["deptreedepth2"],
                "synsim": row1['synsim'],
                'nbchars1': row2['nbchars1'],
                'nbchars2': row2['nbchars2'],
                'levsim': row2['levsim']
            })


def merge_data_further():
    df1 = pd.read_csv('../Predict_RSRS/data/PKU-10w-UnderLen120.csv')
    df1.copy()
    df_merge = df1
    df2 = json.load(open('syndiv.json', 'r', encoding = 'utf-8'))
    syndiv = df2['syndiv']
    para1 = df2['para1']
    for row, syndiversity, para in zip(df1.iterrows(), syndiv, para1):
        rid, row = row
        para = para.replace(' ', '')
        assert row['paraphrase1'] == para
    df_merge['syndiv'] = syndiv
    new_df = pd.DataFrame(columns = ['paraphrase1', 'paraphrase2', 'nbchars1', 'nbchars2',
                                     'lex_diff1', 'lex_diff2', 'lex_sim',
                                     'syn_diff1', 'syn_diff2', 'syn_sim', 'sem_sim'])
    for rid, row in df_merge.iterrows():
        if (len(row['paraphrase1']) <= 120) and (len(row['paraphrase2']) <= 120) and (len(row['paraphrase2']) > 30) and (len(row['paraphrase2']) > 30):
            new_df.loc[len(new_df)] = [
                row['paraphrase1'], row['paraphrase2'], row['nbchars1'], row['nbchars2'],
                row['levdiff1'], row['levdiff2'], row['levsim'],
                row['syndiff1'], row['syndiff2'], 1 - row['syndiv'], row['synsim']
            ]
    new_df.to_csv('PKU-10w-UnderLen120Over30.csv', index = False, sep = ',')


def get_css_qpinput_feature():
    df = pd.read_csv('css_features_qp.csv')
    df.copy()
    df['NbChars'] = df.apply(lambda x: x['nbchars1']/x['nbchars2'], axis = 1)  # 值越大表示压缩越多
    df['LexDiff'] = df.apply(lambda x: x['lex_diff2']/x['lex_diff1'], axis = 1)
    df['SynDiff'] = df.apply(lambda x: x['syn_diff1']/x['syn_diff2'], axis = 1)
    df = df[['complex', 'simple', 'NbChars', 'LexDiff', 'lex_sim', 'SynDiff', 'syn_sim', 'sem_sim']]
    df.to_csv('css_for_qp.csv', index = False, sep = ',')


def get_qp_martin2020features_1():
    df = pd.read_csv('../Predict_RSRS/data/PKU-10w-UnderLen120.csv')
    df.copy()
    df['NbChars'] = df.apply(lambda x: x['nbchars2']/x['nbchars1'], axis = 1)  # 值越大表示压缩越多
    df['LevSim'] = df['lex_sim']
    df['WordRank'] = df.apply(lambda x: x['lex_diff2']/x['lex_diff1'], axis = 1)
    df['DepTreeDepth'] = df.apply(lambda x: x['syn_diff2']/x['syn_diff1'], axis = 1)
    df = df[['paraphrase1', 'paraphrase2', 'NbChars', 'LevSim', 'WordRank', 'DepTreeDepth']]
    df = df.rename(columns = {'paraphrase1': 'source', 'paraphrase2': 'target'})
    NbChars_normal = sigmoid(np.array(df['NbChars'].tolist()))
    LevSim_normal = sigmoid(np.array(df['LevSim'].tolist()))
    WordRank_normal = sigmoid(np.array(df['WordRank'].tolist()))
    DepTreeDepth_normal = sigmoid(np.array(df['DepTreeDepth'].tolist()))
    df['NbChars'] = NbChars_normal
    df['LevSim'] = LevSim_normal
    df['WordRank'] = WordRank_normal
    df['DepTreeDepth'] = DepTreeDepth_normal
    df.to_csv('PKU-10w-UnderLen120_martin2020features.csv', index = False, sep = ',')


def get_qp_martin2020features_1_reverse():
    df = pd.read_csv('../Predict_RSRS/data/PKU-10w-UnderLen120.csv')
    df.copy()
    df['NbChars'] = df.apply(lambda x: x['nbchars1']/x['nbchars2'], axis = 1)  # 值越大表示压缩越多
    df['LevSim'] = df['lex_sim']
    df['WordRank'] = df.apply(lambda x: x['lex_diff1']/x['lex_diff2'], axis = 1)
    df['DepTreeDepth'] = df.apply(lambda x: x['syn_diff1']/x['syn_diff2'], axis = 1)
    df = df[['paraphrase1', 'paraphrase2', 'NbChars', 'LevSim', 'WordRank', 'DepTreeDepth']]
    df = df.rename(columns = {'paraphrase1': 'source', 'paraphrase2': 'target'})
    NbChars_normal = sigmoid(np.array(df['NbChars'].tolist()))
    LevSim_normal = sigmoid(np.array(df['LevSim'].tolist()))
    WordRank_normal = sigmoid(np.array(df['WordRank'].tolist()))
    DepTreeDepth_normal = sigmoid(np.array(df['DepTreeDepth'].tolist()))
    df['NbChars'] = NbChars_normal
    df['LevSim'] = LevSim_normal
    df['WordRank'] = WordRank_normal
    df['DepTreeDepth'] = DepTreeDepth_normal
    df.to_csv('PKU-10w-UnderLen120_martin2020features_1.csv', index = False, sep = ',')


def filter_underLen30():
    df = pd.read_csv('PKU-10w-UnderLen120_martin2020features_1.csv')
    headers = list(df.keys())
    filter_df = pd.DataFrame(columns = headers)
    for rid, row in df.iterrows():
        if len(row['source']) > 30 and len(row['target']) > 30:
            filter_df.loc[len(filter_df)] = [
                row[headers[0]], row[headers[1]], row[headers[2]], row[headers[3]],
                row[headers[4]], row[headers[5]]
            ]
    filter_df.to_csv('PKU-10w-UnderLen120-OverLen30-MartinFeature_1.csv', index = False, sep = ',')


def get_qp_martin2020features_1_noNormalize():
    """
    在1的基础上不使用normalize
    :return:
    """
    df = pd.read_csv('css_features_qp_test.csv')
    df.copy()
    df['NbChars'] = df.apply(lambda x: x['nbchars2']/x['nbchars1'], axis = 1)  # 值越大表示压缩越多
    df['LevSim'] = df['lex_sim']
    df['WordRank'] = df.apply(lambda x: x['lex_diff2']/x['lex_diff1'], axis = 1)
    df['DepTreeDepth'] = df.apply(lambda x: x['syn_diff2']/x['syn_diff1'], axis = 1)
    df = df[['complex', 'simple', 'NbChars', 'LevSim', 'WordRank', 'DepTreeDepth']]
    df = df.rename(columns = {'complex': 'source', 'simple': 'target'})
    # NbChars_normal = sigmoid(np.array(df['NbChars'].tolist()))
    # LevSim_normal = sigmoid(np.array(df['LevSim'].tolist()))
    # WordRank_normal = sigmoid(np.array(df['WordRank'].tolist()))
    # DepTreeDepth_normal = sigmoid(np.array(df['DepTreeDepth'].tolist()))
    df['NbChars'] = np.array(df['NbChars'].tolist())
    df['LevSim'] = np.array(df['LevSim'].tolist())
    df['WordRank'] = np.array(df['WordRank'].tolist())
    df['DepTreeDepth'] = np.array(df['DepTreeDepth'].tolist())
    df.to_csv('css_qp_martin2020features_1_noNormalize_test.csv', index = False, sep = ',')


def feature_normalize():
    df = pd.read_csv('../Pseudo-CSS-QCPG/PKU_css_qp_martinFeature.csv')
    df.copy()
    nbchars_rs = sigmoid(np.array(df['NbChars_rs']))
    levsim_rs = sigmoid(np.array(df['LevSim_rs']))
    wordrank_rs = sigmoid(np.array(df['WordRank_rs']))
    deptreedepth_rs = sigmoid(np.array(df['DepTreeDepth_rs']))
    nbchars_o = sigmoid(np.array(df['NbChars_o']))
    levsim_o = sigmoid(np.array(df['LevSim_o']))
    wordrank_o = sigmoid(np.array(df['WordRank_o']))
    deptreedepth_o = sigmoid(np.array(df['DepTreeDepth_o']))
    df['NbChars_rs'] = nbchars_rs
    df['NbChars_o'] = nbchars_o
    df['LevSim_rs'] = levsim_rs
    df['LevSim_o'] = levsim_o
    df['WordRank_rs'] = wordrank_rs
    df['WordRank_o'] = wordrank_o
    df['DepTreeDepth_rs'] = deptreedepth_rs
    df['DepTreeDepth_o'] = deptreedepth_o
    df.to_csv('../Pseudo-CSS-QCPG/PKU_css_qp_martinFeature_normalize.csv', index = False, sep = ',')


if __name__ == '__main__':
    # merge_data()
    # merge_data_further()
    # get_css_qpinput_feature()
    # get_qp_martin2020features_1_noNormalize()
    # filter_underLen30()
    # feature_normalize()
    print()

