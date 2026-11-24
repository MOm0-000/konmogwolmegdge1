"""SST-2 classification: local-first data, CLS/mean/max pooling comparison."""
import argparse
import csv
import gc
import inspect
import json
import os
import re
import time
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset, DatasetDict, DownloadConfig, load_dataset, load_from_disk
from transformers import (AutoModel, AutoTokenizer, DataCollatorWithPadding,
                          Trainer, TrainingArguments, set_seed)
from BERT import DistilBertForSequenceClassification, POOLING_METHODS

BASE = Path(__file__).resolve().parent
CACHE = BASE / 'cache'
MODEL = CACHE / 'distilbert'
DATASET = 'stanfordnlp/sst2'
PRETRAINED = 'distilbert-base-uncased'
SPLITS = ('train', 'validation')
ID2LABEL = {0: 'NEGATIVE', 1: 'POSITIVE'}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pooling', choices=[*POOLING_METHODS, 'all'], default='cls')
    p.add_argument('--epochs', type=int, default=2)
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--lr', type=float, default=2e-5)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--max_length', type=int, default=128)
    p.add_argument('--fp16', action='store_true')
    p.add_argument('--model_path', type=Path, default=MODEL)
    p.add_argument('--dataset_path', type=Path)
    p.add_argument('--offline', action='store_true')
    a = p.parse_args()
    if a.epochs < 1 or a.batch_size < 1 or a.lr <= 0 or not 2 <= a.max_length <= 512:
        p.error('Invalid training parameters.')
    if a.fp16 and not torch.cuda.is_available():
        p.error('--fp16 requires CUDA.')
    a.model_path = a.model_path.expanduser().resolve()
    if a.dataset_path:
        a.dataset_path = a.dataset_path.expanduser().resolve()
    return a


def config_suffix(a):
    parts = []
    for key, baseline, prefix in [('epochs', 2, 'e'), ('batch_size', 16, 'bs'),
                                  ('lr', 2e-5, 'lr'), ('seed', 42, 'seed'),
                                  ('max_length', 128, 'len')]:
        val = getattr(a, key)
        if val != baseline:
            parts.append(f'{prefix}{val:g}')
    if a.fp16:
        parts.append('fp16')
    return '_'.join(parts) or 'baseline'


def prepare_model(a):
    # Load actual local files, not merely config.json. No remote requests here.
    path = a.model_path
    try:
        if not path.is_dir():
            raise FileNotFoundError(path)
        tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True)
        encoder = AutoModel.from_pretrained(str(path), local_files_only=True)
        del encoder
        print(f'使用本地模型：{path}', flush=True)
        return tokenizer
    except OSError as exc:
        if path != MODEL.resolve():
            raise RuntimeError(f'指定模型目录无法加载：{path}') from exc
    for local_only in ([True] if a.offline else [True, False]):
        try:
            tokenizer = AutoTokenizer.from_pretrained(PRETRAINED, cache_dir=str(CACHE), local_files_only=local_only)
            encoder = AutoModel.from_pretrained(PRETRAINED, cache_dir=str(CACHE), local_files_only=local_only)
            break
        except OSError:
            if a.offline or not local_only:
                raise
    path.mkdir(parents=True, exist_ok=True)
    encoder.save_pretrained(path)
    tokenizer.save_pretrained(path)
    del encoder
    print(f'预训练模型已准备：{path}', flush=True)
    return tokenizer


def validate_dataset(ds):
    for split in SPLITS:
        if split not in ds or not len(ds[split]) or not {'sentence', 'label'}.issubset(ds[split].column_names):
            raise ValueError(f'Invalid SST-2 split: {split}')
        if set(ds[split]['label']) - {0, 1}:
            raise ValueError(f'Invalid sentiment labels: {split}')
        label = ds[split].features['label']
        if hasattr(label, 'names') and [s.lower() for s in label.names] != ['negative', 'positive']:
            raise ValueError(f'Unexpected SST-2 label order: {label.names}')
    return DatasetDict({s: ds[s] for s in SPLITS})


def find_local_dataset(root):
    """Accept saved DatasetDict, original HF Arrow files and Parquet shards."""
    if not root.is_dir():
        return None
    folders = [root] + sorted((p for p in root.rglob('*') if p.is_dir()), key=str)
    for folder in folders:
        if (folder / 'dataset_dict.json').is_file():
            ds = load_from_disk(str(folder))
            return validate_dataset(ds), {'format': 'saved_dataset', 'path': str(folder)}
        for ext in ('arrow', 'parquet'):
            # HF 的 parquet builder 会生成 parquet-train.arrow 等缓存文件。
            for prefix in ('sst2-', 'parquet-', 'glue-', ''):
                files = {s: [folder / f'{prefix}{s}.{ext}'] for s in SPLITS}
                if not all(p.is_file() and p.stat().st_size for ps in files.values() for p in ps):
                    files = {s: sorted(folder.glob(f'{prefix}{s}-*-of-*.{ext}')) for s in SPLITS}
                if not all(files.values()):
                    continue
                # Reject incomplete numbered shard sets.
                complete = True
                for ps in files.values():
                    matches = [re.search(r'-(\d+)-of-(\d+)\.', p.name) for p in ps]
                    if any(matches):
                        if not all(matches) or {int(m[2]) for m in matches} != {len(ps)} or {int(m[1]) for m in matches} != set(range(len(ps))):
                            complete = False
                if not complete:
                    continue
                if ext == 'arrow':
                    from datasets import concatenate_datasets
                    ds = DatasetDict({s: concatenate_datasets([Dataset.from_file(str(p)) for p in ps])
                                      if len(ps) > 1 else Dataset.from_file(str(ps[0])) for s, ps in files.items()})
                else:
                    ds = DatasetDict({s: Dataset.from_parquet([str(p) for p in ps]) for s, ps in files.items()})
                return validate_dataset(ds), {'format': ext, 'files': {s: [str(p) for p in ps] for s, ps in files.items()}}
    return None


def load_sst2(a):
    if a.dataset_path:
        roots = [a.dataset_path]
    else:
        roots = [CACHE / 'sst2_local', CACHE / 'sst2', CACHE / 'stanfordnlp___sst2', CACHE / 'glue' / 'sst2',
                 CACHE / 'datasets--stanfordnlp--sst2' / 'snapshots']
        if CACHE.is_dir():
            roots += [p for p in CACHE.rglob('*') if p.is_dir() and
                      ('sst2' in p.name.lower() or p.name.lower() == 'sst-2')]
    for root in dict.fromkeys(roots):
        found = find_local_dataset(root)
        if found:
            print(f'使用本地 SST-2：{root}', flush=True)
            # prepare_dataset 随后会打印实际文件路径及样本数量。
            return found
    if a.dataset_path:
        raise FileNotFoundError(f'指定目录中缺少完整 SST-2 train/validation 文件：{a.dataset_path}')
    if a.offline or os.environ.get('HF_HUB_OFFLINE', '').upper() in ('1', 'TRUE', 'YES', 'ON'):
        raise FileNotFoundError('离线模式下未找到 SST-2，请通过 --dataset_path 指定本地目录。')
    # First try cached HF dataset, then download if missing.
    try:
        ds = load_dataset(DATASET, cache_dir=str(CACHE),
                          download_config=DownloadConfig(local_files_only=True))
        origin = 'hf_cache'
    except (OSError, ValueError, RuntimeError):
        print('本地 SST-2 缓存缺失，下载数据集。', flush=True)
        ds = load_dataset(DATASET, cache_dir=str(CACHE))
        origin = 'download'
    ds = validate_dataset(ds)
    target = CACHE / 'sst2_local'
    ds.save_to_disk(str(target))
    print(f'SST-2 数据来源：{origin}；已保存本地数据：{target}', flush=True)
    return ds, {'format': 'saved_dataset', 'path': str(target)}


def prepare_dataset(a, tokenizer):
    ds, source = load_sst2(a)
    print(json.dumps(source, ensure_ascii=False), flush=True)
    for s in SPLITS:
        print(f'{s}: {len(ds[s])} sentences', flush=True)
    def preprocess(examples):
        return tokenizer(examples['sentence'], truncation=True, max_length=a.max_length)
    encoded = {s: ds[s].map(preprocess, batched=True,
                          remove_columns=[c for c in ds[s].column_names if c != 'label']) for s in SPLITS}
    return encoded, source


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    if isinstance(logits, tuple):
        logits = logits[0]
    return {'accuracy': float(np.mean(np.argmax(logits, axis=-1) == labels))}


def train_one(a, method, tokenizer, ds, source, suffix):
    name = f'{method}_{suffix}'
    print(f'\n实验名称：{name}', flush=True)
    set_seed(a.seed)
    model = DistilBertForSequenceClassification.from_pretrained(
        str(a.model_path), local_files_only=True, num_labels=2,
        id2label=ID2LABEL, label2id={v: k for k, v in ID2LABEL.items()})
    model.config.pooling = method
    ckpt = BASE / 'ckpt' / f'{name}_ckpt'
    model_dir = BASE / 'ckpt' / f'{name}_model'
    result_dir = BASE / 'result'
    result_dir.mkdir(exist_ok=True)
    arg_params = inspect.signature(TrainingArguments.__init__).parameters
    strategy = 'eval_strategy' if 'eval_strategy' in arg_params else 'evaluation_strategy'
    training_args = TrainingArguments(output_dir=str(ckpt), learning_rate=a.lr,
        per_device_train_batch_size=a.batch_size, per_device_eval_batch_size=a.batch_size * 2,
        num_train_epochs=a.epochs, weight_decay=0.01, **{strategy: 'epoch'}, save_strategy='epoch',
        logging_steps=100, load_best_model_at_end=True, metric_for_best_model='accuracy',
        greater_is_better=True, save_total_limit=1, report_to='none', seed=a.seed,
        data_seed=a.seed, fp16=a.fp16, optim='adamw_torch')
    proc = 'processing_class' if 'processing_class' in inspect.signature(Trainer.__init__).parameters else 'tokenizer'
    trainer = Trainer(model=model, args=training_args, train_dataset=ds['train'], eval_dataset=ds['validation'],
                      **{proc: tokenizer}, data_collator=DataCollatorWithPadding(tokenizer), compute_metrics=compute_metrics)
    start = time.perf_counter()
    train = trainer.train().metrics
    metrics = trainer.evaluate()
    trainer.save_model(str(model_dir))
    tokenizer.save_pretrained(model_dir)
    trainer.save_state()
    config = dict(epochs=a.epochs, batch_size=a.batch_size, eval_batch_size=a.batch_size*2,
                  learning_rate=a.lr, weight_decay=0.01, optimizer='adamw_torch', seed=a.seed,
                  max_length=a.max_length, fp16=a.fp16, model_path=str(a.model_path),
                  train_size=len(ds['train']), validation_size=len(ds['validation']))
    result = dict(pooling=method, run_name=name, config=config, dataset='SST-2', dataset_source=source,
                  accuracy=float(metrics['eval_accuracy']), eval_loss=float(metrics['eval_loss']),
                  train=train, validation=metrics, best_checkpoint=trainer.state.best_model_checkpoint,
                  total_seconds=time.perf_counter()-start)
    if method == 'cls':
        result.update(target_accuracy=0.88, target_reached=result['accuracy'] > 0.88)
    (result_dir / f'{name}_summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    rows = [r for r in trainer.state.log_history if 'eval_accuracy' in r]
    with (result_dir / f'{name}_history.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['epoch', 'step', 'eval_loss', 'eval_accuracy'], extrasaction='ignore')
        writer.writeheader(); writer.writerows(rows)
    print(f'{method}: accuracy={result["accuracy"]:.6f}, loss={result["eval_loss"]:.6f}', flush=True)
    if method == 'cls':
        print('CLS accuracy > 0.88:', result['target_reached'], flush=True)
    del trainer, model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def save_comparison(suffix, current):
    root = BASE / 'result'
    rows = []
    for method in POOLING_METHODS:
        p = root / f'{method}_{suffix}_summary.json'
        if p.is_file():
            r = json.loads(p.read_text(encoding='utf-8'))
            if r['config'] != current['config'] or r['dataset_source'] != current['dataset_source']:
                print(f'跳过配置或数据来源不同的旧结果：{p.name}', flush=True)
                continue
            rows.append(dict(pooling=method, accuracy=r['accuracy'], eval_loss=r['eval_loss'],
                             train_loss=r['train']['train_loss'], train_seconds=r['train']['train_runtime'],
                             **r['config']))
    if not rows:
        return
    path = root / ('pooling_comparison.csv' if suffix == 'baseline' else f'pooling_comparison_{suffix}.csv')
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    print(f'\n对比结果：{path}', flush=True)
    for row in rows:
        print(f'{row["pooling"]:<5} Accuracy={row["accuracy"]:.6f} Loss={row["eval_loss"]:.6f}', flush=True)


def main():
    a = parse_args()
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    print('Device:', 'CUDA' if torch.cuda.is_available() else 'CPU', flush=True)
    tokenizer = prepare_model(a)
    ds, source = prepare_dataset(a, tokenizer)
    suffix = config_suffix(a)
    for method in (POOLING_METHODS if a.pooling == 'all' else (a.pooling,)):
        current = train_one(a, method, tokenizer, ds, source, suffix)
        save_comparison(suffix, current)


if __name__ == '__main__':
    main()
