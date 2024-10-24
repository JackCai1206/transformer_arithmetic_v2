from transformers import EvalPrediction, PreTrainedTokenizer, TrainerCallback, Trainer
from transformers.integrations import WandbCallback
import Levenshtein
import numpy as np

from lib.configs import MyTrainingArguments
from typing import Union, Tuple, Optional


def get_real_label(input_str):
    a, b = input_str.split('=')[0].split('+')
    a = a.split('C')[1]
    a = a[::-1]
    b = b[::-1]
    c = str(int(a) + int(b))
    c = c[::-1]
    c = c.ljust(max(len(a), len(b))+1, '0')

    return c

def compute_metrics(tokenizer: PreTrainedTokenizer, pred_obj: EvalPrediction, args: MyTrainingArguments):
    # import ipdb; ipdb.set_trace()
    '''
    pred_obj: 
        predictions: Union[np.ndarray, Tuple[np.ndarray]],
        label_ids: Union[np.ndarray, Tuple[np.ndarray]],
        inputs: Optional[Union[np.ndarray, Tuple[np.ndarray]]] = None,
    '''
    pred = pred_obj.predictions[:, pred_obj.inputs.shape[1]:]
    labels = pred_obj.label_ids
    # Padding handled in the trainer predict
    # min_len = min(pred.shape[1], labels.shape[1])
    # pred = pred[:, :min_len]
    # labels = labels[:, :min_len]
    
    if args.do_backtrack_decoding or args.do_backtrack_eval or args.do_backtrack_decoding2:
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

    # Print debug information
    prompt_str = tokenizer.batch_decode(pred_obj.inputs)
    pred_str = tokenizer.batch_decode(pred_obj.predictions[:, pred_obj.inputs.shape[1]:], skip_special_tokens=True)
    label_str = tokenizer.batch_decode(pred_obj.label_ids)
    
    metrics = {
        'accuracy': accuracy,
        'distance': distance
    }
    if args.get_real_label:
        real_label_str = [get_real_label(pr) for pr in prompt_str]
        reaL_acc = [real_label_str[i] == pred_str[i] for i in range(len(real_label_str))]
        reaL_acc = sum(reaL_acc) / len(reaL_acc)

        metrics['real_accuracy'] = reaL_acc
    
    prompt_str = prompt_str[:5]
    pred_str = pred_str[:5]
    label_str = label_str[:5]

    for i, (pr, p, l) in enumerate(zip(prompt_str, pred_str, label_str)):
        print("="*80)
        print(f"Prompt     : {repr(pr)}")
        print(f"Pred       : {repr(p)}")
        print(f"Label      : {repr(l)}")
        if args.get_real_label:
            print(f"Real Label : {repr(real_label_str[i])}")


    return metrics


def compute_metrics_new(tokenizer: PreTrainedTokenizer, pred_obj: EvalPrediction, args: MyTrainingArguments):
    pred = pred_obj.predictions[:, pred_obj.inputs.shape[1]:]
    labels = pred_obj.label_ids
    
    # Initialize tracking variables
    first_wrong_locs = []
    backtrack_counts = []
    first_backtrack_locs = []
    
    # For backtrack decoding cases
    if args.do_backtrack_decoding or args.do_backtrack_eval or args.do_backtrack_decoding2:
        # Get the backtrack token ID
        backtrack_token_id = tokenizer.backtrack_token_id
        cleaned_pred = np.zeros_like(pred)
        
        for bi, p in enumerate(pred):
            delete_mask = p == backtrack_token_id
            delete_mask[:-1] |= delete_mask[1:]
            p_cleaned = p[~delete_mask]
            cleaned_pred[bi, :len(p_cleaned)] = p_cleaned
            
            # Track number of backtrack tokens
            backtrack_count = np.sum(p == backtrack_token_id)
            backtrack_counts.append(backtrack_count)
            
            # Track the first occurrence of a backtrack token
            first_backtrack_loc = np.where(p == backtrack_token_id)[0]
            if len(first_backtrack_loc) > 0:
                first_backtrack_locs.append(first_backtrack_loc[0] + 1)  # +1 for 1-based index
            else:
                first_backtrack_locs.append(None)  # If no backtrack tokens are found
        
        pred = cleaned_pred[:, :labels.shape[1]]
    
    for bi, p in enumerate(pred):
        # Track the first wrong prediction location
        first_wrong_loc = np.where(p != labels[bi])[0]
        if len(first_wrong_loc) > 0:
            first_wrong_locs.append(first_wrong_loc[0] + 1)  # +1 for 1-based index
        else:
            first_wrong_locs.append(None)  # If no wrong predictions are found

    # Calculate accuracy and distance metrics
    accuracy = (pred == labels).all(axis=1).mean()
    distance = sum([Levenshtein.ratio(pred[bi].tolist(), labels[bi].tolist()) for bi in range(pred.shape[0])]) / pred.shape[0]
    
    # Decode and print the predictions
    prompt_str = tokenizer.batch_decode(pred_obj.inputs[:5])
    pred_str = tokenizer.batch_decode(pred_obj.predictions[:5, pred_obj.inputs.shape[1]:])
    cleaned_pred_str = tokenizer.batch_decode(pred[:5])
    label_str = tokenizer.batch_decode(pred_obj.label_ids[:5])
    for pr, p, cp, l in zip(prompt_str, pred_str, cleaned_pred_str, label_str):
        print("="*80)
        print(f"Prompt: {repr(pr)}")
        print(f"Pred  : {repr(p)}")
        print(f"Cleaned Pred  : {repr(cp)}")
        print(f"Label : {repr(l)}")
    
    # Calculate average metrics
    avg_first_wrong_loc = np.mean([loc for loc in first_wrong_locs if loc is not None]) if first_wrong_locs else None
    avg_first_backtrack_loc = np.mean([loc for loc in first_backtrack_locs if loc is not None]) if first_backtrack_locs else None
    avg_backtrack_count = np.mean(backtrack_counts) if backtrack_counts else 0
    
    # Prepare the results dictionary
    metrics = {
        'accuracy': accuracy,
        'distance': distance,
        'avg_first_wrong_loc': avg_first_wrong_loc,
        'avg_backtrack_count': avg_backtrack_count,
        'avg_first_backtrack_loc': avg_first_backtrack_loc
    }
    
    return metrics



class WandbEvalCallback(WandbCallback):
    def __init__(self, trainer: Trainer):
        super().__init__()

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        breakpoint()
