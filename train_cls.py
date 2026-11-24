from BERT import DistilBertForSequenceClassification
from transformers import AutoTokenizer
from datasets import load_dataset
import numpy as np
import evaluate
from datasets import DownloadMode

id2label = {0: "NEGATIVE", 1: "POSITIVE"}
label2id = {"NEGATIVE": 0, "POSITIVE": 1}

tokenizer = AutoTokenizer.from_pretrained("./cache/distilbert")
model = DistilBertForSequenceClassification.from_pretrained("./cache/distilbert", num_labels=2, id2label=id2label, label2id=label2id)

dataset = load_dataset("stanfordnlp/sst2", cache_dir="./cache", download_mode=DownloadMode.REUSE_DATASET_IF_EXISTS)
accuracy = evaluate.load("accuracy", cache_dir="./cache")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    res = accuracy.compute(predictions=predictions, references=labels)
    print(res)
    return res

