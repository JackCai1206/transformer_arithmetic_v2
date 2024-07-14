from transformers import EvalPrediction, PreTrainedTokenizer, TrainerCallback
from transformers.integrations import WandbCallback

def compute_metrics(tokenizer: PreTrainedTokenizer, pred_obj: EvalPrediction):
    pred = pred_obj.predictions[:, pred_obj.inputs.shape[1]:]
    labels = pred_obj.label_ids
    min_len = min(pred.shape[1], labels.shape[1])
    pred = pred[:, :min_len]
    labels = labels[:, :min_len]
    accuracy = (pred == labels).all(axis=1).mean()

    prompt_str = tokenizer.batch_decode(pred_obj.inputs[:5])
    pred_str = tokenizer.batch_decode(pred_obj.predictions[:5, pred_obj.inputs.shape[1]:])
    label_str = tokenizer.batch_decode(pred_obj.label_ids[:5])
    for pr, p, l in zip(prompt_str, pred_str, label_str):
        print("="*80)
        print(f"Prompt: {repr(pr)}")
        print(f"Pred  : {repr(p)}")
        print(f"Label : {repr(l)}")

    return {'accuracy': accuracy}

# class LogEvalCallback(WandbCallback):
#     def __init__(self, tokenizer: PreTrainedTokenizer):
#         super().__init__()
#         self.tokenizer = tokenizer

#     def on_evaluate(self, args, state, control, metrics):
#         breakpoint()
