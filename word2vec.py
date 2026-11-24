"""训练 CBOW / Skip-Gram，并保存模型及训练记录。"""
import argparse
from pathlib import Path
from time import perf_counter

from gensim.models import word2vec

BASE = Path(__file__).resolve().parent
DEFAULT_CORPUS = BASE / 'data/seg_text.txt'


def experiment_name(corpus, sg):
    return f'{Path(corpus).stem}_sg{sg}'


def main():
    parser = argparse.ArgumentParser(description='Word2Vec 训练')
    parser.add_argument('--sg', type=int, choices=[0, 1], default=0,
                        help='0=CBOW, 1=Skip-Gram')
    parser.add_argument('--corpus', type=Path, default=DEFAULT_CORPUS)
    parser.add_argument('--window', type=int, default=1)
    parser.add_argument('--vector_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--workers', type=int, default=1)
    args = parser.parse_args()
    name = experiment_name(args.corpus, args.sg)
    model_path = BASE / 'ckpt' / f'{name}.model'
    log_path = BASE / 'result' / f'{name}_train.txt'
    model_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    method = 'CBOW' if args.sg == 0 else 'Skip-Gram'
    with log_path.open('w', encoding='utf-8') as log:
        def report(message):
            print(message, flush=True)
            print(message, file=log, flush=True)

        report(f'语料：{args.corpus}')
        report(f'开始训练：{method}, sg={args.sg}, window={args.window}, '
               f'vector_size={args.vector_size}, epochs={args.epochs}, '
               f'seed={args.seed}, workers={args.workers}')
        start = perf_counter()
        model = word2vec.Word2Vec(
            word2vec.Text8Corpus(str(args.corpus)),
            window=args.window, vector_size=args.vector_size,
            epochs=args.epochs, seed=args.seed, sg=args.sg,
            workers=args.workers,
        )
        report(f'训练完成，耗时 {perf_counter() - start:.2f} 秒，词表大小 {len(model.wv)}')
        # 原始语料的验收指标；新语料通过 play.py --pair 检测。
        if args.corpus.resolve() == DEFAULT_CORPUS.resolve():
            scores = []
            for word in ('林平之', '岳灵珊', '葵花宝典', '易筋经'):
                if '岳不群' in model.wv and word in model.wv:
                    score = float(model.wv.similarity('岳不群', word))
                    report(f'sim(岳不群, {word}) = {score:.8f}')
                    if word in ('林平之', '岳灵珊'):
                        scores.append(score)
                else:
                    report(f'词表缺少：岳不群或{word}')
            report(f'两对相似度均 > 0.95：{len(scores) == 2 and all(s > 0.95 for s in scores)}')
        model.save(str(model_path))
        report(f'模型：{model_path}')
        report(f'训练记录：{log_path}')


if __name__ == '__main__':
    main()
