from datetime import datetime
from pathlib import Path
from shutil import copy2
from time import perf_counter

from gensim.models import word2vec

# 当前语料的实测配置：两对指定人物的余弦相似度均大于 0.95。
# 更大的窗口或更多训练轮数，不一定让指定词对的相似度更高。
window = 1        # 上下文窗口：只考虑紧邻的词
vector_size = 64  # 词向量维度
sg = 0          # 1 为 Skip-gram，0 为 CBOW
epochs = 3        # 训练轮数
seed = 42         # 随机种子
workers = 1       # 单线程，避免多线程更新顺序带来的随机波动


def main():
    base_dir = Path(__file__).resolve().parent
    sentences = word2vec.Text8Corpus(str(base_dir / "data" / "seg_text.txt"))
    print(
        f"开始训练：sg={sg}, window={window}, vector_size={vector_size}, "
        f"epochs={epochs}, seed={seed}, workers={workers}",
        flush=True,
    )
    start = perf_counter()
    model = word2vec.Word2Vec(
        sentences,
        window=window,
        vector_size=vector_size,
        epochs=epochs,
        seed=seed,
        sg=sg,
        workers=workers,
    )
    print(f"训练完成，耗时 {perf_counter() - start:.2f} 秒，词表大小 {len(model.wv)}")

    target_scores = []
    for word in ("林平之", "岳灵珊"):
        score = float(model.wv.similarity("岳不群", word))
        target_scores.append(score)
        print(f"sim(岳不群, {word}) = {score:.8f}")
    print("两对相似度均 > 0.95：", all(score > 0.95 for score in target_scores))

    # 同时查看不同类别词的相似度，避免只看两个人物词对就判断模型质量。
    for word in ("葵花宝典", "易筋经"):
        print(f"对照 sim(岳不群, {word}) = {model.wv.similarity('岳不群', word):.8f}")

    model_path = base_dir / "ckpt" / "1019.model"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    if model_path.exists():
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        backup_path = model_path.with_name(f"1019_backup_{timestamp}.model")
        copy2(model_path, backup_path)
        print(f"已备份旧模型：{backup_path}")
    model.save(str(model_path))
    print(f"已保存新模型：{model_path}")


if __name__ == "__main__":
    main()
