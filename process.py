"""中文语料编码识别与 jieba 分词。"""
import argparse
import codecs
from pathlib import Path

import chardet
import jieba

BASE = Path(__file__).resolve().parent


def read_corpus(path, encoding=None):
    raw = path.read_bytes()
    if not encoding:
        # 优先识别 BOM，再尝试 UTF-8，最后检测其他编码。
        for bom, name in ((codecs.BOM_UTF32_LE, 'utf-32'),
                          (codecs.BOM_UTF32_BE, 'utf-32'),
                          (codecs.BOM_UTF8, 'utf-8-sig'),
                          (codecs.BOM_UTF16_LE, 'utf-16'),
                          (codecs.BOM_UTF16_BE, 'utf-16')):
            if raw.startswith(bom):
                encoding = name
                break
        if not encoding:
            try:
                return raw.decode('utf-8'), 'utf-8'
            except UnicodeDecodeError:
                detected = chardet.detect(raw)
                encoding = detected['encoding']
                if not encoding:
                    raise ValueError('编码检测失败，请通过 --encoding 指定编码。')
                print(f"编码检测置信度：{detected['confidence']:.2%}")
                if encoding.lower() in {'gb2312', 'gbk'}:
                    encoding = 'gb18030'
    try:
        return raw.decode(encoding), encoding
    except UnicodeDecodeError as exc:
        raise ValueError(f'无法按 {encoding} 解码，请通过 --encoding 指定正确编码。') from exc


def main():
    parser = argparse.ArgumentParser(description='自动识别编码并进行中文分词')
    parser.add_argument('--input', type=Path, default=BASE / 'data/new_text.txt')
    parser.add_argument('--output', type=Path, default=None)
    parser.add_argument('--encoding', default=None, help='手动指定编码（默认自动检测）')
    args = parser.parse_args()
    source = args.input
    # 默认 text.txt -> seg_text.txt，new_text.txt -> new_seg_text.txt。
    target = args.output
    if target is None:
        name = source.stem
        stem = name[:-4] + 'seg_text' if name.endswith('text') else name + '_seg'
        target = source.with_name(stem + '.txt')
    if source.resolve() == target.resolve():
        parser.error('输入文件与输出文件应使用不同路径。')
    text, encoding = read_corpus(source, args.encoding)
    target.parent.mkdir(parents=True, exist_ok=True)
    # w 模式保证重复执行时不会累积旧语料。
    count = 0
    with target.open('w', encoding='utf-8') as f:
        for line in text.splitlines():
            tokens = [word for word in jieba.cut(line) if word.strip()]
            if tokens:
                f.write(' '.join(tokens) + '\n')
                count += len(tokens)
    print(f'输入：{source}\n编码：{encoding}\n输出：{target}\n词元数量：{count}')


if __name__ == '__main__':
    main()
