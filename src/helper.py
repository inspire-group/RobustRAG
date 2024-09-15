import json 
import random
import numpy as np
import torch

def get_log_name(args):
    if args.defense_method == 'none': defense_str = 'none'
    elif args.defense_method == 'voting': defense_str = f'{args.defense_method}'
    elif args.defense_method == 'keyword': defense_str = f'{args.defense_method}-{args.alpha}-{args.beta}' 
    elif args.defense_method == 'decoding': defense_str = f'{args.defense_method}-{args.eta}-{args.subsample_iter}' 
    else: defense_str = ""
    return f'{args.dataset_name}-{args.model_name}-{defense_str}-top{args.top_k}-corr{args.corruption_size}-attack{args.attack_method}'


def load_jsonl(file_path):
    results = []
    with open(file_path, 'r') as f:
        for line in f:  # This avoids loading all lines into memory at once
            results.append(json.loads(line))
    return results


def save_json(file,file_path):
    with open(file_path, 'w') as g:
        g.write(json.dumps(file, indent=4))

def load_json(file_path):
    with open(file_path) as file:
        results = json.load(file)
    return results

def setup_seeds(seed):
    # seed = config.run_cfg.seed + get_rank()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def clean_str(s):
    try:
        s=str(s)
    except:
        print('Error: the output cannot be converted to a string')
    s=s.strip()
    #if len(s)>1 and s[-1] == ".":
    #    s=s[:-1]
    return s.lower()

def f1_score(precision, recall):
    """
    Calculate the F1 score given precision and recall arrays.
    
    Args:
    precision (np.array): A 2D array of precision values.
    recall (np.array): A 2D array of recall values.
    
    Returns:
    np.array: A 2D array of F1 scores.
    """
    f1_scores = np.divide(2 * precision * recall, precision + recall, where=(precision + recall) != 0)
    
    return f1_scores






from transformers import StoppingCriteriaList, StoppingCriteria
from torch import LongTensor, FloatTensor, eq, device




class StopOnTokens(StoppingCriteria):
    #https://discuss.huggingface.co/t/implimentation-of-stopping-criteria-list/20040/13
    def __init__(self, stop_token_ids):
        """
        Initializes the stopping criteria with a list of token ID lists,
        each representing a sequence of tokens that should cause generation to stop.
        
        :param stop_token_ids: List of lists of token IDs
        """
        self.stop_token_ids = stop_token_ids

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        """
        Check if any of the stop token sequences are at the end of the current sequence of input IDs.

        :param input_ids: torch.LongTensor representing the sequence of token IDs generated so far
        :param scores: torch.FloatTensor representing the generation scores (unused in this criterion)
        :return: True if stopping condition met, False otherwise
        """
        # Iterate over each set of stop token IDs
        for stop_ids in self.stop_token_ids:
            if eq(input_ids[0][-len(stop_ids[0])+1:], stop_ids[0][1:]).all():
                return True
        return False