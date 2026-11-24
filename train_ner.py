"""DistilBERT + CoNLL2003 NER experiment."""
import argparse
import csv
import json
import inspect
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset, DatasetDict, load_dataset
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import LocalEntryNotFoundError
from seqeval.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import (AutoModel, AutoModelForTokenClassification, AutoTokenizer,
                          DataCollatorForTokenClassification, Trainer, TrainingArguments, set_seed)

BASE = Path(__file__).resolve().parent
CACHE = BASE / 'cache'
MODEL = CACHE / 'distilbert'
DATASET = 'eriktks/conll2003'
REVISION = 'e8237f8875e4ecf7f60b07ae49691b85cf893410'
LABELS = ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC']
TARGETS = dict(precision=0.87, recall=0.89, f1=0.88, accuracy=0.975)


def automatic_run_name(a):
    """Default settings form the baseline; changed training settings name the run."""
    parts = []
    if a.epochs != 1:
        parts.append(f'e{a.epochs}')
    if a.batch_size != 8:
        parts.append(f'bs{a.batch_size}')
    if a.lr != 2e-5:
        parts.append(f'lr{a.lr}')
    if a.seed != 42:
        parts.append(f'seed{a.seed}')
    if a.max_length != 512:
        parts.append(f'len{a.max_length}')
    if a.fp16:
        parts.append('fp16')
    return 'ner_' + ('_'.join(parts) if parts else 'baseline')


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--epochs', type=int, default=1)
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--lr', type=float, default=2e-5)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--max_length', type=int, default=512)
    p.add_argument('--fp16', action='store_true')
    p.add_argument('--run_name', default=None, help='可选自定义实验名；默认根据训练参数自动命名')
    p.add_argument('--model_path', default=str(MODEL), help='本地预训练模型目录')
    p.add_argument('--dataset_path', type=Path, default=None, help='本地 Arrow/Parquet 数据集目录；默认自动搜索 cache/conll2003')
    p.add_argument('--offline', action='store_true', help='只使用本地文件及缓存')
    a = p.parse_args()
    if min(a.epochs, a.batch_size) < 1 or a.lr <= 0 or not 2 <= a.max_length <= 512:
        p.error('Check epochs, batch_size, lr and max_length.')
    if a.fp16 and not torch.cuda.is_available():
        p.error('--fp16 requires CUDA.')
    if a.run_name is None:
        a.run_name = automatic_run_name(a)
    if not a.run_name or Path(a.run_name).name != a.run_name or a.run_name in ('.', '..'):
        p.error('run_name must be a simple filename.')
    return a


def prepare_model(a):
    path = Path(a.model_path).expanduser().resolve()
    model_kwargs = dict(num_labels=len(LABELS), id2label=dict(enumerate(LABELS)),
                        label2id={label: i for i, label in enumerate(LABELS)})

    # 直接读取本地目录；local_files_only=True 保证此步骤不访问网络。
    try:
        tokenizer = AutoTokenizer.from_pretrained(str(path), use_fast=True, local_files_only=True)
        model = AutoModelForTokenClassification.from_pretrained(
            str(path), local_files_only=True, **model_kwargs)
        print(f'使用本地模型：{path}', flush=True)
        return tokenizer, model
    except OSError as exc:
        if path != MODEL.resolve():
            raise RuntimeError(f'指定的本地模型目录缺少可加载的文件：{path}') from exc
        print(f'默认模型目录缺少可加载的文件：{path}', flush=True)

    # 默认目录缺失时，先检查 Hugging Face 模型缓存，再按需联网下载。
    for local_only in ([True] if a.offline else [True, False]):
        try:
            tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased',
                use_fast=True, cache_dir=str(CACHE), local_files_only=local_only)
            encoder = AutoModel.from_pretrained('distilbert-base-uncased',
                cache_dir=str(CACHE), local_files_only=local_only)
            break
        except OSError as exc:
            if a.offline or not local_only:
                raise RuntimeError('无法加载 DistilBERT 预训练模型，请检查本地缓存或网络。') from exc
    path.mkdir(parents=True, exist_ok=True)
    encoder.save_pretrained(path)
    tokenizer.save_pretrained(path)
    del encoder
    model = AutoModelForTokenClassification.from_pretrained(
        str(path), local_files_only=True, **model_kwargs)
    print(f'预训练模型已准备：{path}', flush=True)
    return tokenizer, model


SPLITS = ('train', 'validation', 'test')


def find_local_dataset(root):
    """Find three splits in the same directory; accept HF Arrow caches or Parquet."""
    if not root.is_dir():
        return None
    # 首选直接指定的目录，其次搜索 Hugging Face 的版本/哈希子目录。
    dirs = [root] + sorted((d for d in root.rglob('*') if d.is_dir()),
                           key=lambda d: (-d.stat().st_mtime, str(d)))
    for folder in dirs:
        for extension in ('arrow', 'parquet'):
            for prefix in ('conll2003-', ''):
                files = {split: folder / f'{prefix}{split}.{extension}' for split in SPLITS}
                if all(p.is_file() and p.stat().st_size > 0 for p in files.values()):
                    return extension, files
    return None


def load_ner_dataset(a):
    root = a.dataset_path.expanduser().resolve() if a.dataset_path else CACHE / 'conll2003'
    found = find_local_dataset(root)
    if found:
        extension, files = found
        print(f'使用本地 {extension.upper()} 数据集：{next(iter(files.values())).parent}', flush=True)
        if extension == 'arrow':
            dataset = DatasetDict({split: Dataset.from_file(str(path)) for split, path in files.items()})
        else:
            dataset = load_dataset('parquet', data_files={s: str(p) for s, p in files.items()},
                                   cache_dir=str(CACHE))
        return dataset, {'format': extension, 'files': {s: str(p) for s, p in files.items()}}
    if a.dataset_path is not None:
        raise FileNotFoundError(f'指定目录缺少完整的 train/validation/test 数据集：{root}')

    # 没有原始本地数据时，逐个分片检查 Hub 缓存，只有缺失的分片才联网下载。
    files = {}
    for split in SPLITS:
        kwargs = dict(repo_id=DATASET, repo_type='dataset', revision=REVISION,
                      filename=f'conll2003/conll2003-{split}.parquet', cache_dir=str(CACHE))
        try:
            files[split] = hf_hub_download(**kwargs, local_files_only=True)
        except LocalEntryNotFoundError as exc:
            if a.offline:
                raise FileNotFoundError(f'离线模式：缺少 {split} 数据集缓存。') from exc
            print(f'下载缺失的数据集分片：{split}', flush=True)
            files[split] = hf_hub_download(**kwargs)
    dataset = load_dataset('parquet', data_files=files, cache_dir=str(CACHE))
    return dataset, {'format': 'parquet', 'files': files, 'repo': DATASET, 'revision': REVISION}


def prepare_dataset(tokenizer, a):
    dataset, source = load_ner_dataset(a)
    max_length = a.max_length
    for split in SPLITS:
        ds = dataset[split]
        if not {'tokens', 'ner_tags'}.issubset(ds.column_names) or not len(ds):
            raise ValueError(f'Invalid CoNLL2003 split: {split}')
        feature = ds.features['ner_tags'].feature
        if hasattr(feature, 'names') and list(feature.names) != LABELS:
            raise ValueError(f'Unexpected NER label order: {split}: {feature.names}')
        print(f'{split}: {len(ds)} sentences', flush=True)

    def align(examples):
        encoded = tokenizer(examples['tokens'], is_split_into_words=True,
                            truncation=True, max_length=max_length)
        labels = []
        for i, tags in enumerate(examples['ner_tags']):
            previous = None
            row = []
            for word_id in encoded.word_ids(batch_index=i):
                row.append(tags[word_id] if word_id is not None and word_id != previous else -100)
                previous = word_id
            labels.append(row)
        encoded['labels'] = labels
        return encoded

    return {split: ds.map(align, batched=True, remove_columns=ds.column_names)
            for split, ds in dataset.items()}, source


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    if isinstance(logits, tuple):
        logits = logits[0]
    predictions = np.argmax(logits, axis=-1)
    pred = [[LABELS[p] for p, y in zip(ps, ys) if y != -100] for ps, ys in zip(predictions, labels)]
    gold = [[LABELS[y] for y in ys if y != -100] for ys in labels]
    return dict(precision=precision_score(gold, pred), recall=recall_score(gold, pred),
                f1=f1_score(gold, pred), accuracy=accuracy_score(gold, pred))


def main():
    a = parse_args()
    print(f'实验名称：{a.run_name}', flush=True)
    set_seed(a.seed)
    print('Device:', 'CUDA' if torch.cuda.is_available() else 'CPU', flush=True)
    tokenizer, model = prepare_model(a)
    ds, source = prepare_dataset(tokenizer, a)
    result_dir = BASE / 'result'
    result_dir.mkdir(exist_ok=True)
    # Adapt to Transformers 4.x and 5.x without changing the training hyperparameters.
    arg_params = inspect.signature(TrainingArguments.__init__).parameters
    strategy_key = 'eval_strategy' if 'eval_strategy' in arg_params else 'evaluation_strategy'
    args = TrainingArguments(
        output_dir=str(BASE / 'ckpt' / f'{a.run_name}_ckpt'),
        learning_rate=a.lr,
        per_device_train_batch_size=a.batch_size, per_device_eval_batch_size=a.batch_size,
        num_train_epochs=a.epochs, weight_decay=0.01, **{strategy_key: 'epoch'},
        save_strategy='epoch', logging_steps=50, load_best_model_at_end=True,
        metric_for_best_model='f1', greater_is_better=True, save_total_limit=1,
        report_to='none', seed=a.seed, data_seed=a.seed, fp16=a.fp16,
        optim='adamw_torch',
    )
    trainer_params = inspect.signature(Trainer.__init__).parameters
    processor_key = 'processing_class' if 'processing_class' in trainer_params else 'tokenizer'
    trainer = Trainer(model=model, args=args, train_dataset=ds['train'], eval_dataset=ds['validation'],
                      **{processor_key: tokenizer}, data_collator=DataCollatorForTokenClassification(tokenizer),
                      compute_metrics=compute_metrics)
    train_metrics = trainer.train().metrics
    # Model selection is on validation; report test metrics only after training.
    validation = trainer.evaluate()
    test = trainer.evaluate(ds['test'], metric_key_prefix='test')
    passed = {key: float(test[f'test_{key}']) > threshold for key, threshold in TARGETS.items()}
    model_dir = BASE / 'ckpt' / f'{a.run_name}_model'
    trainer.save_model(str(model_dir))
    tokenizer.save_pretrained(model_dir)
    config = vars(a).copy()
    if config['dataset_path'] is not None:
        config['dataset_path'] = str(config['dataset_path'])
    config.update(eval_batch_size=a.batch_size, weight_decay=0.01, optimizer='adamw_torch',
                  selection_split='validation', final_eval_split='test')
    summary = dict(config=config, dataset='CoNLL2003', dataset_source=source,
                   best_checkpoint=trainer.state.best_model_checkpoint,
                   train=train_metrics, validation=validation, test=test,
                   targets=TARGETS, passed=passed, all_passed=all(passed.values()))
    (result_dir / f'{a.run_name}_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    with (result_dir / f'{a.run_name}_metrics.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['metric', 'validation', 'test', 'target_strictly_greater_than', 'passed'])
        for key, threshold in TARGETS.items():
            writer.writerow([key, validation[f'eval_{key}'], test[f'test_{key}'], threshold, passed[key]])
    rows = [x for x in trainer.state.log_history if 'eval_f1' in x]
    if rows:
        fields = ['epoch', 'eval_loss', 'eval_precision', 'eval_recall', 'eval_f1', 'eval_accuracy']
        with (result_dir / f'{a.run_name}_history.csv').open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print('All test thresholds passed:', all(passed.values()))


if __name__ == '__main__':
    main()
