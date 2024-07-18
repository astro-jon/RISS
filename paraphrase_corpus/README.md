## paraphrase corpus readme

- [data](data):dir saving corus 
    - (data download url: https://huggingface.co/datasets/ZzZzzO0o/riss_paraphrase/tree/main)
    - [pku_features_50w_add-distilbert-multilingual.csv](data%2Fpku_features_50w_add-distilbert-multilingual.csv): pku paraphrase corpus after computing feature value

| column_name         | feature_meaning                                                          |
|---------------------|--------------------------------------------------------------------------|
| nbchars1/2          | number of sentence                                                       |
| lex_diff1/2         | wordrank of sentence                                                     |
| lex_sim             | the Levenshtein similarity between pairwise sentence (⭐used in our work) |
| syn_diff1/2         | the depth of syntatic dependency tree                                    |
| syn_sim             | the similarity of syntatic tree   (⭐used in our work)                                         |
| sem_sim             | the similarity of semantic between pairwise sentence  (⭐used in our work)                     | 
| rsrs1/2             | the RSRS value scored by multilingual-bert                               |
| rsrs1/2 distil bert | the RSRS value scored by distil multilingual-bert  (⭐used in our work)   |

  - [0_compute_similarity.py](0_compute_similarity.py): compute 3 similarity (levsim, synsim, semsim)
  - [1_compute_rsrs.py](1_compute_rsrs.py): compute readability
  - [2_RPS_filter.py](2_RPS_filter.py): according to relationship between language features and readability differences to mine high-quality corpus
  - 