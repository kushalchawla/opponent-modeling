from transformers import AutoTokenizer
import os
BASE_STORAGE = os.environ['OppModelingStorage']

"""
Using the tokenizer from the transformer (Huggingface) framework
"""

BASE_TOKENS = {

    "additional_special_tokens": [
        "<self>",
        "<opponent>",
        'firewood', 
        'hydrated', 
        'campfire',
    ],
}

def get_special_tokens_dict():
    return BASE_TOKENS

def load_tokenizer(pretrained="bert", special_token_dict=None):
    """
    THIS IS NEVER CALLED FROM THE CODE.
    Tokenizer was created offline using this code, then stored under *_common_tokenizer, and then always supplied with the argument - to ensure common tokenizer for all runs of the code, either training or evaluation.
    """
    tokenizer = None

    if(pretrained == 'bert'):
        tokenizer = AutoTokenizer.from_pretrained(f"bert-base-uncased", use_fast=False)
    elif(pretrained == 'roberta'):
        tokenizer = AutoTokenizer.from_pretrained(f"roberta-base", use_fast=False)
    else:
        raise NotImplementedError
    
    assert tokenizer

    if special_token_dict is None:
        special_token_dict = BASE_TOKENS
    num_added_toks = tokenizer.add_special_tokens(special_token_dict)
    print(f"Extended {pretrained} tokenizer with {num_added_toks}")
    for special_token in tokenizer.additional_special_tokens:
        print("\t" + special_token)
    return tokenizer