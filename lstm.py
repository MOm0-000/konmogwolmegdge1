"""单层 LSTM 词级语言模型实验。"""
import argparse
import csv
import json
import math
import random
from collections import Counter
from pathlib import Path
from time import perf_counter

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

BASE = Path(__file__).resolve().parent


class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size, num_layers):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        out, _ = self.lstm(self.embedding(x))
        return self.fc(out)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--corpus', type=Path, default=BASE / 'data/text.txt')
    p.add_argument('--segmented', action='store_true', help='输入已经分词，以空白分隔')
    p.add_argument('--encoding', default='auto')
    p.add_argument('--vocab_size', type=int, default=3000)
    p.add_argument('--embed_dim', type=int, default=64)
    p.add_argument('--hidden_size', type=int, default=256)
    p.add_argument('--num_layers', type=int, default=1)
    p.add_argument('--seq_len', type=int, default=31)
    p.add_argument('--stride', type=int, default=None, help='默认等于窗口长度')
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--lr', type=float, default=0.002)
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--target_loss', type=float, default=3.5)
    p.add_argument('--val_ratio', type=float, default=0.05)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    p.add_argument('--run_name', default='lstm')
    a = p.parse_args()
    a.stride = a.seq_len if a.stride is None else a.stride
    if min(a.embed_dim, a.hidden_size, a.num_layers, a.seq_len, a.stride, a.batch_size, a.epochs) < 1 or a.vocab_size < 2:
        p.error('模型维度、窗口、批次和轮数必须为正整数，词表至少为 2')
    if not 0 < a.val_ratio < 1 or a.lr <= 0 or a.target_loss <= 0:
        p.error('请检查 val_ratio、lr 和 target_loss')
    if Path(a.run_name).name != a.run_name or a.run_name in ('', '.', '..'):
        p.error('run_name 必须是简单文件名')
    return a


def read_tokens(path, encoding, segmented):
    raw = path.read_bytes()
    if encoding == 'auto':
        if raw.startswith((b'\xff\xfe\x00\x00', b'\x00\x00\xfe\xff')):
            encoding = 'utf-32'
        elif raw.startswith((b'\xff\xfe', b'\xfe\xff')):
            encoding = 'utf-16'
        else:
            for candidate in ('utf-8-sig', 'gb18030'):
                try:
                    raw.decode(candidate)
                    encoding = candidate
                    break
                except UnicodeDecodeError:
                    pass
            else:
                import chardet
                encoding = chardet.detect(raw)['encoding']
                if not encoding:
                    raise ValueError('编码识别失败，请使用 --encoding 指定')
    text = raw.decode(encoding)
    print(f'语料：{path}，编码：{encoding}', flush=True)
    if segmented:
        tokens = text.split()
    else:
        import jieba
        tokens = [w for w in jieba.cut(text) if w.strip()]
    return tokens


def build_vocab(tokens, vocab_size):
    words = ['<UNK>'] + [w for w, _ in Counter(tokens).most_common() if w != '<UNK>'][:vocab_size - 1]
    return words, {w: i for i, w in enumerate(words)}


def make_loader(tokens, word2id, args, shuffle, device):
    ids = torch.tensor([word2id.get(w, 0) for w in tokens], dtype=torch.long)
    if len(ids) <= args.seq_len:
        raise ValueError('语料分段过短，无法构造语言模型窗口')
    unk_rate = (ids == 0).float().mean().item()
    windows = ids.unfold(0, args.seq_len + 1, args.stride)
    # 验证使用不重叠窗口，避免重复统计相同位置。
    if not shuffle:
        windows = ids.unfold(0, args.seq_len + 1, args.seq_len)
    ds = TensorDataset(windows[:, :-1], windows[:, 1:])
    return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle,
                      num_workers=0, pin_memory=device.type == 'cuda'), unk_rate


def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss, total_tokens = 0.0, 0
    with torch.set_grad_enabled(training):
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            if training:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            total_loss += loss.item() * y.numel()
            total_tokens += y.numel()
    return total_loss / total_tokens


def save_curve(history, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot([r['epoch'] for r in history], [r['train_loss'] for r in history], 'o-', label='Train')
    ax.plot([r['epoch'] for r in history], [r['val_loss'] for r in history], 's-', label='Validation')
    ax.axhline(history[0]['target_loss'], color='gray', linestyle='--', label='Target')
    ax.set(xlabel='Epoch', ylabel='CrossEntropyLoss', title='LSTM Language Modeling')
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA 不可用，请检查 PyTorch 安装或使用 --device cpu')
    print(f'训练设备：{device}', flush=True)
    if device.type == 'cuda':
        print(torch.cuda.get_device_name(0), flush=True)
    tokens = read_tokens(args.corpus, args.encoding, args.segmented)
    split = int(len(tokens) * (1 - args.val_ratio))
    train_tokens, val_tokens = tokens[:split], tokens[split:]
    # 按原始文本顺序切分；仅使用训练集建立词表。
    words, word2id = build_vocab(train_tokens, args.vocab_size)
    train_loader, train_unk = make_loader(train_tokens, word2id, args, True, device)
    val_loader, val_unk = make_loader(val_tokens, word2id, args, False, device)
    print(f'总词数：{len(tokens)}，实际词表：{len(words)}，训练/验证窗口：{len(train_loader.dataset)}/{len(val_loader.dataset)}', flush=True)
    print(f'UNK 比例：训练 {train_unk:.2%}，验证 {val_unk:.2%}', flush=True)
    config = dict(vocab_size=len(words), embed_dim=args.embed_dim, hidden_size=args.hidden_size, num_layers=args.num_layers)
    model = LSTMModel(**config).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    ckpt, result = BASE / 'ckpt', BASE / 'result'
    ckpt.mkdir(exist_ok=True)
    result.mkdir(exist_ok=True)
    name = args.run_name
    settings = vars(args).copy()
    settings['corpus'] = str(args.corpus)
    settings.update(actual_vocab_size=len(words), train_unk_rate=train_unk, val_unk_rate=val_unk, device_used=str(device))
    (result / f'{name}_config.json').write_text(json.dumps(settings, ensure_ascii=False, indent=2), encoding='utf-8')
    fields = ['epoch', 'train_loss', 'val_loss', 'val_perplexity', 'seconds', 'target_loss']
    history, best = [], float('inf')
    start = perf_counter()
    with (result / f'{name}_loss.csv').open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for epoch in range(1, args.epochs + 1):
            t = perf_counter()
            train_loss = run_epoch(model, train_loader, criterion, device, optimizer)
            val_loss = run_epoch(model, val_loader, criterion, device)
            row = dict(epoch=epoch, train_loss=train_loss, val_loss=val_loss,
                       val_perplexity=math.exp(min(val_loss, 700)), seconds=perf_counter()-t, target_loss=args.target_loss)
            history.append(row)
            writer.writerow(row)
            f.flush()
            print(f'Epoch {epoch:02d}/{args.epochs} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_ppl={row["val_perplexity"]:.2f} | {row["seconds"]:.1f}s', flush=True)
            checkpoint = dict(model_state_dict=model.state_dict(), model_config=config,
                              word2id=word2id, id2word=words, epoch=epoch,
                              train_loss=train_loss, val_loss=val_loss, args=settings)
            torch.save(checkpoint, ckpt / f'{name}_last.pt')
            if val_loss < best:
                best = val_loss
                torch.save(checkpoint, ckpt / f'{name}_best.pt')
            if train_loss < args.target_loss:
                print(f'训练损失达到目标：{train_loss:.4f} < {args.target_loss}', flush=True)
                break
    save_curve(history, result / f'{name}_loss.png')
    summary = dict(epochs_completed=len(history), final_train_loss=history[-1]['train_loss'],
                   final_val_loss=history[-1]['val_loss'], best_val_loss=best,
                   target_reached=history[-1]['train_loss'] < args.target_loss,
                   total_seconds=perf_counter()-start)
    (result / f'{name}_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(f'模型保存在 {ckpt}，训练记录保存在 {result}', flush=True)


if __name__ == '__main__':
    main()
