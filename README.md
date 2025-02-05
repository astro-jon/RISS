# RISS 
📚We will continue to improve the project......

## 😃Model available
you can download our pretrained model from: https://huggingface.co/ZzZzzO0o/riss

## 😁Project structure
- [paraphrase_corpus](paraphrase_corpus): paraphrase corpus pre-process serving as training data
- [simplify](simplify): model training scripts
  - base model, we select bart-base-chinese
  - [data](simplify%2Fdata): containing mined data by RPS, CSS, MCTS, and idiom-paraphrase corpus

## 🤗Raw data download
- CSS: https://github.com/maybenotime/CSS
- MCTS: https://github.com/blcuicall/mcts
- idiom-paraphrase: https://github.com/jpqiang/Chinese-Idiom-Paraphrasing

## Contact
If you find any problems or have any questions, please contact the author by email: audbut0702@163.com.


## Cite

如果 RISS 对您的研究有启发，请您引用：

```
@misc{zhang-2024-riss,
      title={Readability-guided Idiom-aware Sentence Simplification (RISS) for Chinese}, 
      author={Jingshen Zhang and Xinglu Chen and Xinying Qiu and Zhimin Wang and Wenhe Feng},
      year={2024},
      eprint={2406.02974},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```