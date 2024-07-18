import numpy
import pandas as pd
import numpy as np
from tqdm import tqdm
from itertools import chain
import seaborn as sns
from matplotlib import rcParams
import matplotlib.pyplot as plt
import scipy.stats as stats


def min_max_y(raw_data):
    min_max_data = []
    min_num = min(raw_data)
    max_num = max(raw_data)
    # Min–max normalization
    for d in tqdm(raw_data):
        min_max_data.append((d - min_num) / (max_num - min_num))

    return min_max_data


def get_mean_std(data):
    return np.mean(data), np.std(data)


def standardize(data):
    dataMean = np.mean(data)
    dataStd = np.std(data)
    standData = [(i - dataMean) / dataStd for i in data]
    return standData, np.mean(standData), np.std(standData)


def xigema_filter(df):
    global LEX_MEAN, SYN_MEAN, SEM_MEAN
    lexsimList, lexsimMean, lexsimStd = standardize(df["lex_sim"].tolist())
    synsimList, synsimMean, synsimStd = standardize(df["syn_sim"].tolist())
    semsimList, semsimMean, semsimStd = standardize(df["sem_sim"].tolist())
    LEX_MEAN.append(np.mean(df["lex_sim"].tolist()))
    SYN_MEAN.append(np.mean(df["syn_sim"].tolist()))
    SEM_MEAN.append(np.mean(df["sem_sim"].tolist()))
    lexsimCheck = [1 if (lex >= lexsimMean-1*lexsimStd) and (lex <= lexsimMean+1*lexsimStd) else 0 for lex in lexsimList]
    synsimCheck = [1 if (syn >= synsimMean-1*synsimStd) and (syn <= synsimMean+1*synsimStd) else 0 for syn in synsimList]
    semsimCheck = [1 if (sem >= semsimMean - 1 * semsimStd) and (sem <= semsimMean + 1 * semsimStd) else 0 for sem in semsimList]
    df["lexsim check"] = lexsimCheck
    df["synsim check"] = synsimCheck
    df["semsim check"] = semsimCheck
    return df


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def Visual_UpdateConfig():
    sns.set(style = "darkgrid")
    config = {
        "font.family": "serif",
        "font.size": 3,
        "mathtext.fontset": 'stix',
        "axes.unicode_minus": False,
        "xtick.direction": 'in',
        "ytick.direction": 'in'
    }
    rcParams.update(config)


def visual_distribution(data, tilte, bin = 10):
    databin = [bin*round(d*100/bin) for d in data]
    xlist = [i/100 for i in range(min(databin), max(databin) + bin, bin)]
    ylist = [databin.count(int(i*100))/len(databin) for i in xlist]
    plt.plot(xlist, ylist, "#E68E70", linewidth = 1.2, label = "RSRS Difference")
    plt.legend()
    plt.xlabel("RSRS Difference in Paraphrase-PKU")
    plt.ylabel("Percentages")
    plt.title(tilte)
    plt.savefig(f"../visual/{tilte}.svg", format = "svg", dpi = 600)
    plt.show()


def featrue_rs_relation(rsrs_difference, lexlist, synlist, semlist):
    lexrel = stats.spearmanr(rsrs_difference, lexlist)
    synrel = stats.spearmanr(rsrs_difference, synlist)
    semrel = stats.spearmanr(rsrs_difference, semlist)
    print(f"Lexical:{lexrel}\nSyntactic:{synrel}\nSematic:{semrel}\n\n")


def logNormal(data):
    return numpy.log10(data)


if __name__ == '__main__':
    Visual_UpdateConfig()
    df = pd.read_csv("data/pku_features_50w_add-distilbert-multilingual.csv")
    rsrsdiff = [abs(r1-r2) if (not np.isinf(abs(r1-r2))) and (not np.isnan(abs(r1 - r2))) else 0
                for r1, r2 in zip(df["rsrs1 distil"].tolist(), df["rsrs2 distil"].tolist())]
    df["rsrs Difference"] = rsrsdiff
    df = df.loc[(df["rsrs Difference"] > 0)].reset_index(drop = True)  # 498959 → 498897

    # binning filter paraphrase-3-factor
    df["rsrs Difference"] = logNormal(np.array(df["rsrs Difference"].tolist()))

    rsrsdiff = df["rsrs Difference"].tolist()
    rsrs_difference_bin = [5 * round(rd * 100 / 5) for rd in rsrsdiff]
    df["rsrs Difference bin"] = rsrs_difference_bin
    LEX_MEAN, SYN_MEAN, SEM_MEAN = [], [], []
    binlist = sorted(list(set(rsrs_difference_bin)))
    df = df.groupby(["rsrs Difference bin"]).apply(lambda x: xigema_filter(x))
    df["flag"] = df["lexsim check"] + df["synsim check"] + df["semsim check"]
    featrue_rs_relation(binlist, LEX_MEAN, SYN_MEAN, SEM_MEAN)
    original_dataNum = len(df)
    df = df.loc[df["flag"] == 3]  # 498897 → 227618
    print(f"Original Num is {original_dataNum}\n"
          f"After filter Num is {len(df)}\n"
          f"Utilisation is {round(len(df) / original_dataNum, 5)}")
    df = df.loc[(df["rsrs Difference"] >= -1)].reset_index(drop = True)  # 227618 → 222638

    sourceList, targetList = [], []
    for rid, row in df.iterrows():
        para1 = row["paraphrase1"]
        para2 = row["paraphrase2"]
        rsrs1 = row["rsrs1 distil"]
        rsrs2 = row["rsrs2 distil"]
        # if row["rsrs Difference bin"] == max(rsrs_difference_bin): continue
        if row["nbchars1"] < 30 or row["nbchars2"] < 30: continue
        if rsrs1 > rsrs2:
            sourceList.append(para1)
            targetList.append(para2)
        else:
            sourceList.append(para2)
            targetList.append(para1)
    dfSave = pd.DataFrame({
        "source": sourceList, "target": targetList
    })  # 64703
    print(len(dfSave))  #

    dfSave.to_csv("data/train.csv", index = False, sep = ",")





