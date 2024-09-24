from datasets import load_dataset, Dataset
print("="*20)
#load_dataset("/home/zhangtianning/.cache/huggingface/datasets/timdettmers___openassistant-guanaco/")
#load_dataset("glue", "mrpc")
load_dataset("/home/zhangtianning/.cache/huggingface/datasets/glue/mrpc")
from transformers import (
    AutoModelForSequenceClassification,
)
import evaluate
metric = evaluate.load("glue", "mrpc")
# model = AutoModelForSequenceClassification.from_pretrained(
#         "bert-base-cased", return_dict=True
#     )
