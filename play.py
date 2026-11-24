"""分析词向量，自动保存相似度、类比结果和 PCA 图片。"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')  # 自动保存图片，无需桌面窗口。
from matplotlib import pyplot as plt
from matplotlib import font_manager
from sklearn.decomposition import PCA
from gensim.models import Word2Vec

from word2vec import BASE, DEFAULT_CORPUS, experiment_name


def configure_font():
    installed = {font.name for font in font_manager.fontManager.ttflist}
    for name in ('Microsoft YaHei', 'Noto Sans CJK SC', 'WenQuanYi Micro Hei',
                 'SimHei', 'PingFang SC', 'Source Han Sans SC'):
        if name in installed:
            matplotlib.rcParams['font.family'] = name
            matplotlib.rcParams['axes.unicode_minus'] = False
            return name
    return None


def main():
    parser = argparse.ArgumentParser(description='Word2Vec 结果分析')
    parser.add_argument('--sg', type=int, choices=[0, 1], default=0)
    parser.add_argument('--corpus', type=Path, default=DEFAULT_CORPUS)
    parser.add_argument('--pair', nargs=2, action='append', metavar=('WORD1', 'WORD2'),
                        help='自定义词对，可重复指定')
    args = parser.parse_args()
    name = experiment_name(args.corpus, args.sg)
    model_path = BASE / 'ckpt' / f'{name}.model'
    model = Word2Vec.load(str(model_path))
    if model.sg != args.sg:
        raise ValueError(f'模型实际 sg={model.sg}，与 --sg {args.sg} 不符。')
    result_dir = BASE / 'result'
    result_dir.mkdir(parents=True, exist_ok=True)
    log_path = result_dir / f'{name}_analysis.txt'
    original = args.corpus.resolve() == DEFAULT_CORPUS.resolve()
    with log_path.open('w', encoding='utf-8') as log:
        def report(message):
            print(message)
            print(message, file=log, flush=True)

        report(f'模型：{model_path}')
        report(f'sg={model.sg}, window={model.window}, vector_size={model.vector_size}, '
               f'epochs={model.epochs}')

        def similarity(a, b):
            missing = [word for word in (a, b) if word not in model.wv]
            if missing:
                report(f'sim({a}, {b})：词表缺少 {", ".join(missing)}')
                return None
            score = float(model.wv.similarity(a, b))
            report(f'sim({a}, {b}) = {score:.8f}')
            return score

        if original:
            report('\n-- 与<岳不群>最相似的10个词 --')
            if '岳不群' in model.wv:
                for word, score in model.wv.most_similar('岳不群', topn=10):
                    report(f'{word} {score:.8f}')
            else:
                report('词表缺少：岳不群')

            report('\n-- 指定词对相似度 --')
            scores = [similarity('岳不群', word) for word in ('林平之', '岳灵珊')]
            report(f'两对相似度均 > 0.95：{all(s is not None and s > 0.95 for s in scores)}')

            report('\n-- 和<岳不群+令狐冲-岳夫人>最相似的词 --')
            positive, negative = ['岳不群', '令狐冲'], ['岳夫人']
            missing = [w for w in positive + negative if w not in model.wv]
            if missing:
                report(f'词表缺少：{", ".join(missing)}')
            else:
                for word, score in model.wv.most_similar(positive=positive, negative=negative, topn=10):
                    report(f'{word} {score:.8f}')

            human = ['岳不群', '岳灵珊', '岳夫人', '林平之']
            kongfu = ['广陵散', '葵花宝典', '易筋经', '紫霞']
            groups = [(human, '人物', 'o'), (kongfu, '武功', '^')]
            items = [w for words, _, _ in groups for w in words if w in model.wv]
            if len(items) >= 2 and model.vector_size >= 2:
                font = configure_font()
                if font is None:
                    report('提示：建议安装 fonts-noto-cjk 以正常显示中文。')
                coords = PCA(n_components=2).fit_transform([model.wv[w] for w in items])
                fig, ax = plt.subplots(figsize=(8, 6))
                for words, label, marker in groups:
                    indexes = [i for i, word in enumerate(items) if word in words]
                    if indexes:
                        ax.scatter(coords[indexes, 0], coords[indexes, 1], marker=marker, label=label)
                for word, (x, y) in zip(items, coords):
                    ax.annotate(word, (x, y), xytext=(4, 4), textcoords='offset points')
                method = 'CBOW' if model.sg == 0 else 'Skip-Gram'
                ax.set(title=f'{method} (sg={model.sg})', xlabel='PC1', ylabel='PC2')
                ax.legend()
                ax.margins(0.15)
                fig.tight_layout()
                image_path = result_dir / f'{name}_human_kongfu.png'
                fig.savefig(image_path, dpi=200, bbox_inches='tight')
                plt.close(fig)
                report(f'图片：{image_path}')
            else:
                report('可用词向量不足，无法进行二维 PCA。')

        if args.pair:
            report('\n-- 自定义词对相似度 --')
            for a, b in args.pair:
                similarity(a, b)
        elif not original:
            report('新语料请通过 --pair 词1 词2 指定需要测试的词对。')
        report(f'分析记录：{log_path}')


if __name__ == '__main__':
    main()
