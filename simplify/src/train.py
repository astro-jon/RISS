import logging
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Optional
from easse.sari import corpus_sari, corpus_sari_Add_SubOpe
# Here to have a nice missing dependency error message early on
import numpy as np
from datasets import load_dataset, load_metric
import pandas as pd
import jieba
import transformers
from filelock import FileLock
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)
from transformers.file_utils import is_offline_mode
from transformers.trainer_utils import get_last_checkpoint, is_main_process
from transformers.utils import check_min_version
import json
from config import DatasetArguments, prepare_dataset, ModelArguments, DataTrainingArguments
from utils import *

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.6.0")

logger = logging.getLogger(__name__)


def normalize_condition(cond, cls):
    cond = '_'.join(['COND', str(cls), str(cond)])
    return cond.replace(' ', '_').upper()


def parse_args():
    parser = HfArgumentParser((ModelArguments, DatasetArguments, DataTrainingArguments, Seq2SeqTrainingArguments))
    script_name = os.getcwd() + '\\train.py'
    model_name_or_path = 'your/path/to/bart-base-chinese'
    output_dir = '../model_output/CSS'
    train_file = '../data/train.csv'
    validation_file = '../data/css/test.csv'
    sys.argv = [
        script_name,
        '--model_name_or_path', model_name_or_path,
        '--do_train',
        '--do_eval',
        '--source_column', 'source',
        '--target_column', 'target',
        '--per_device_train_batch_size', '4',
        '--per_device_eval_batch_size', '4',
        '--predict_with_generate',
        '--evaluation_strategy', 'steps',
        '--num_train_epochs', '6',
        '--lr_scheduler_type', 'constant',
        '--save_total_limit', '1',
        '--dataset_generate_mode', 'force_redownload',
        '--dataset_keep_in_memory',
        '--overwrite_output_dir',
        '--output_dir', output_dir,
        '--dataset_generate_mode', 'force_redownload',
        '--train_file', train_file,
        '--validation_file', validation_file,
        '--prediction', 'false'
    ]
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, dataset_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, dataset_args, data_args, training_args = parser.parse_args_into_dataclasses()
    training_args.save_steps = 1000000  # 保存的步长
    training_args.generation_max_length = 512
    training_args.learning_rate = 2e-5
    training_args.gradient_accumulation_steps = 2
    training_args.load_best_model_at_end = True
    training_args.greater_is_better = True
    training_args.metric_for_best_model = 'sari'
    data_args.source_prefix = ''
    data_args.max_source_length = 256
    data_args.max_target_length = 256
    training_args.logging_steps = 1000000
    training_args.eval_steps = 1000
    training_args.max_steps = 50000
    return model_args, dataset_args, data_args, training_args


def main():

    # from datasets import set_caching_enabled
    # set_caching_enabled(False)

    model_args, dataset_args, data_args, training_args = parse_args()
    logger_writer.write(f"\n\n{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}\t"
                        f"==========> 学习率:{training_args.learning_rate}, 累积梯度:{training_args.gradient_accumulation_steps}, 训练批次:{training_args.num_train_epochs}")
    logger_writer.write(f"\n{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}\t"
                        f"==========> 模型保存路径:{training_args.output_dir}")
    training_args.run_name = 'bart-base-chinese-baseline'
    data_args.run_tags = ['bart-base-chinese', str(training_args.learning_rate), str(training_args.gradient_accumulation_steps)]
    if training_args.run_name:
        import clearml
        task = clearml.Task.init(
            project_name="Chinese_Sentence_Simplification",
            task_name=f"{training_args.run_name}",
            tags=data_args.run_tags
        )
        task.set_resource_monitor_iteration_timeout(0)
        parse_args()

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO if is_main_process(training_args.local_rank) else logging.WARN)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
    logger.info(f"Training/evaluation parameters {training_args}")

    # Set seed before initializing model.
    set_seed(training_args.seed)

    datasets = prepare_dataset(dataset_args, logger)

    conditions_columns = data_args.conditions_columns
    if conditions_columns is not None:
        conditions_columns = json.loads(conditions_columns)
        all_conditions = []
        for column in conditions_columns:
            # conditions = datasets['train'].unique(column)
            # conditions_val = datasets['validation'].unique(column)  # 将验证集的特殊标识符也添加上去
            # conditions = list(set(conditions + conditions_val))
            conditions = [normalize_condition(c, column) for c in range(20, 205, 5)]
            all_conditions += conditions
        all_conditions = sorted(all_conditions)
        logger.info(f"Full conditions list: {all_conditions}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        extra_ids=0,
        additional_special_tokens=all_conditions if conditions_columns else [],
    )
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    if model_args.config_name is not None:
        model = AutoModelForSeq2SeqLM.from_config(config)
    elif model_args.model_name_or_path is not None:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_args.model_name_or_path)
    else:
        raise ValueError("You must specify model_name_or_path or config_name")
    model.resize_token_embeddings(len(tokenizer))

    if model.config.decoder_start_token_id is None:
        raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

    prefix = data_args.source_prefix if data_args.source_prefix is not None else ""

    if training_args.do_train:
        column_names = datasets["train"].column_names
    elif training_args.do_eval:
        column_names = datasets["validation"].column_names
    elif training_args.do_predict:
        column_names = datasets["test"].column_names
    else:
        logger.info("There is nothing to do. Please pass `do_train`, `do_eval` and/or `do_predict`.")
        return

    # Get the column names for input/target.
    if data_args.source_column is None:
        source_column = column_names[0]
    else:
        source_column = data_args.source_column
        if source_column not in column_names:
            raise ValueError(
                f"--source_column' value '{data_args.source_column}' needs to be one of: {', '.join(column_names)}"
            )
    if data_args.target_column is None:
        target_column = column_names[1]
    else:
        target_column = data_args.target_column
        if target_column not in column_names:
            raise ValueError(
                f"--target_column' value '{data_args.target_column}' needs to be one of: {', '.join(column_names)}"
            )

    # Temporarily set max_target_length for training.
    max_target_length = data_args.max_target_length
    padding = "max_length" if data_args.pad_to_max_length else False

    if training_args.label_smoothing_factor > 0 and not hasattr(model, "prepare_decoder_input_ids_from_labels"):
        logger.warning(
            "label_smoothing is enabled but the `prepare_decoder_input_ids_from_labels` method is not defined for"
            f"`{model.__class__.__name__}`. This will lead to loss being calculated twice and will take up more memory"
        )

    def preprocess_function(examples):
        inputs = examples[source_column]
        targets = examples[target_column]

        model_inputs = tokenizer(inputs, max_length=data_args.max_source_length, padding=padding, truncation=True)

        # Setup the tokenizer for targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(targets, max_length=max_target_length, padding=padding, truncation=True)

        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        if padding == "max_length" and data_args.ignore_pad_token_for_loss:
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    if training_args.do_train:
        if "train" not in datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = datasets["train"]
        train_dataset = train_dataset.map(
            preprocess_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
        )

    if training_args.do_eval:
        max_target_length = data_args.val_max_target_length
        if "validation" not in datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = datasets["validation"]
        eval_dataset = eval_dataset.map(
            preprocess_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
        )

    if training_args.do_predict:
        max_target_length = data_args.val_max_target_length
        if "test" not in datasets:
            raise ValueError("--do_predict requires a test dataset")
        predict_dataset = datasets["test"]
        predict_dataset = predict_dataset.map(
            preprocess_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
        )

    # Data collator
    label_pad_token_id = -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8 if training_args.fp16 else None,
    )

    # Metric
    # metric = load_metric("sacrebleu", experiment_id=os.getpid())

    val_df = pd.read_csv("../data/test_for_eval.csv")
    orig_sents_list, refs_sents_list = [], [[], []]
    for rid in range(383):
        orig_sents_list.append(val_df.loc[rid]["source"])
        refs_sents_list[0].append(val_df.loc[rid]["target"])
        refs_sents_list[1].append(val_df.loc[383 + rid]["target"])
    # hsk_vocab = read_hsk_vocab('../metrics/vocab.xls')

    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [[label.strip()] for label in labels]

        return preds, labels

    def compute_metrics(eval_preds):
        global epoch_time, result_writer, logger_writer, BEST_Char, BEST_Word, step_time
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        # if data_args.ignore_pad_token_for_loss:
        #     labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        sys_sents_list = [sent.replace(" ", "") for sent in decoded_preds]

        sari_char = []
        for idx, orig in enumerate(orig_sents_list):
            orig_sents_char = " ".join([token for token in orig])
            sys_sents_char = " ".join([token for token in sys_sents_list[idx]])
            refs_sents_char0 = " ".join([token for token in refs_sents_list[0][idx]])
            refs_sents_char1 = " ".join([token for token in refs_sents_list[1][idx]])
            tmp_sari = corpus_sari(
                orig_sents = [orig_sents_char], sys_sents = [sys_sents_char],
                refs_sents = [[refs_sents_char0], [refs_sents_char1]]
            )
            sari_char.append(tmp_sari)
        sari_char = np.mean(sari_char)
        if sari_char > BEST_Char: BEST_Char = sari_char

        sari_word = []
        for idx, orig in enumerate(orig_sents_list):
            orig_sents_word = " ".join(jieba.lcut(orig))
            sys_sents_word = " ".join(jieba.lcut(sys_sents_list[idx]))
            refs_sents_word0 = " ".join(jieba.lcut(refs_sents_list[0][idx]))
            refs_sents_word1 = " ".join(jieba.lcut(refs_sents_list[1][idx]))
            tmp_sari = corpus_sari(
                orig_sents = [orig_sents_word], sys_sents = [sys_sents_word],
                refs_sents = [[refs_sents_word0], [refs_sents_word1]]
            )
            sari_word.append(tmp_sari)
        sari_word = np.mean(sari_word)
        if sari_word > BEST_Word: BEST_Word = sari_word

        df = pd.DataFrame({
            "orig_sents": orig_sents_list*2,
            "sys_sents": sys_sents_list*2,
            "refs_sents": refs_sents_list[0] + refs_sents_list[1]
        })
        df.to_csv(f'../output/model_pred_simp/result/result_{step_time}.csv', index = False, sep = ',')

        result = {"bleu": 0.0, "sari": sari_char, "sari_word": sari_word,
                  "sari_char_max_mean": BEST_Char, "sari_word_max_mean": BEST_Word}

        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        result = {k: round(v, 4) for k, v in result.items()}
        logger_writer.write(f"\n{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}\t"
                            f"==========> Step-Time{step_time}: {result}")
        epoch_time += 1
        step_time += 1000
        result_writer.write(json.dumps({
            "sari_char": sari_char, "sari_word": sari_word,
            "best_sari_char": BEST_Char, "best_sari_word": BEST_Word
        }) + "\n")
        return result

    # Initialize our Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics if training_args.predict_with_generate else None,
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics
        metrics["train_samples"] = len(train_dataset)

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    results = {}
    # if training_args.do_eval:
    #     logger.info("*** Evaluate ***")
    #
    #     metrics = trainer.evaluate(
    #         max_length=data_args.val_max_target_length, num_beams=data_args.num_beams, metric_key_prefix="eval"
    #     )
    #     metrics["eval_samples"] = len(eval_dataset)
    #
    #     trainer.log_metrics("eval", metrics)
    #     trainer.save_metrics("eval", metrics)

    if training_args.do_predict:
        logger.info("*** Predict ***")

        predict_results = trainer.predict(
            predict_dataset,
            metric_key_prefix="predict",
            max_length=data_args.val_max_target_length,
            num_beams=data_args.num_beams,
        )
        metrics = predict_results.metrics
        metrics["predict_samples"] = len(predict_dataset)

        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)

        if trainer.is_world_process_zero():
            if training_args.predict_with_generate:
                predictions = tokenizer.batch_decode(
                    predict_results.predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )
                predictions = [pred.strip() for pred in predictions]
                output_prediction_file = os.path.join(training_args.output_dir, data_args.output_file_name)
                with open(output_prediction_file, "w") as writer:
                    writer.write("\n".join(predictions))

    return results


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    epoch_time = 1
    BEST_Char = -1
    BEST_Word = -1
    step_time = 1000
    os.makedirs('../output/model_pred_simp/result', exist_ok = True)
    logger_writer = open('../output/log.txt', 'a+', encoding = 'utf-8')
    result_writer = open('../output/model_pred_simp/result/res.jsonlines', 'a+', encoding = "utf-8")

    main()

