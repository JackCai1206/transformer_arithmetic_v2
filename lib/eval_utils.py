from transformers import EvalPrediction, PreTrainedTokenizer, TrainerCallback
from transformers.integrations import WandbCallback

def compute_metrics(tokenizer: PreTrainedTokenizer, pred: EvalPrediction):
    accuracy = (pred.predictions[:, pred.inputs.shape[1]:] == pred.label_ids).all(axis=1).mean()

    prompt_str = tokenizer.batch_decode(pred.inputs[:10])
    pred_str = tokenizer.batch_decode(pred.predictions[:10, pred.inputs.shape[1]:])
    label_str = tokenizer.batch_decode(pred.label_ids[:10])
    for pr, p, l in zip(prompt_str, pred_str, label_str):
        print(f"Prompt: {repr(pr)}")
        print(f"Pred: {repr(p)}")
        print(f"Label: {repr(l)}")

    return {'accuracy': accuracy}

# class LogEvalCallback(WandbCallback):
#     def __init__(self, tokenizer: PreTrainedTokenizer):
#         super().__init__()
#         self.tokenizer = tokenizer

#     def on_evaluate(self, args, state, control, metrics):
#         breakpoint()
