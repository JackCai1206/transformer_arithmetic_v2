from transformers import EvalPrediction, PreTrainedTokenizer, TrainerCallback, Trainer
from transformers.integrations import WandbCallback
import Levenshtein
import numpy as np

from lib.trainer_utils import MyTrainingArguments

def compute_metrics(tokenizer: PreTrainedTokenizer, pred_obj: EvalPrediction, args: MyTrainingArguments):
    pred = pred_obj.predictions[:, pred_obj.inputs.shape[1]:]
    labels = pred_obj.label_ids
    # Padding handled in the trainer predict
    # min_len = min(pred.shape[1], labels.shape[1])
    # pred = pred[:, :min_len]
    # labels = labels[:, :min_len]
    
    if args.do_backtrack_decoding or args.do_backtrack_eval:
        # we need to clean the backtrack tokens
        backtrack_token_id = tokenizer.backtrack_token_id
        cleaned_pred = np.zeros_like(pred)
        for bi, p in enumerate(pred):
            delete_mask = p == backtrack_token_id
            delete_mask[:-1] |= delete_mask[1:]
            p = p[~delete_mask]
            cleaned_pred[bi, :len(p)] = p
        pred = cleaned_pred[:, :labels.shape[1]]

    accuracy = (pred == labels).all(axis=1).mean()
    distance = sum([Levenshtein.ratio(pred[bi].tolist(), labels[bi].tolist()) for bi in range(pred.shape[0])]) / pred.shape[0]

    prompt_str = tokenizer.batch_decode(pred_obj.inputs[:5])
    pred_str = tokenizer.batch_decode(pred_obj.predictions[:5, pred_obj.inputs.shape[1]:])
    label_str = tokenizer.batch_decode(pred_obj.label_ids[:5])
    for pr, p, l in zip(prompt_str, pred_str, label_str):
        print("="*80)
        print(f"Prompt: {repr(pr)}")
        print(f"Pred  : {repr(p)}")
        print(f"Label : {repr(l)}")

    return {'accuracy': accuracy, 'distance': distance}

class WandbEvalCallback(WandbCallback):
    def __init__(self, trainer: Trainer):
        super().__init__()

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        breakpoint()
