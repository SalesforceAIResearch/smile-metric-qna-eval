import time
import json
import bert_score
import numpy as np
from tqdm import tqdm
from rouge_score import rouge_scorer

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# specific to METEOR Implementation
import nltk
from nltk.translate.meteor_score import meteor_score
from nltk.tokenize import word_tokenize
nltk.download('punkt_tab')
nltk.download('wordnet')

def compute_rouge_score(metrics:list=['rougeL'], pred_col='pred', sub_metrics=['fmeasure'], ref_data=None):
    """
    Computes ROUGE scores between reference and candidate sentences.

    Parameters:
        metrics (list): List of ROUGE metrics to compute (e.g., ['rouge1', 'rouge2','rougeL']).
        pred_col (str): Name of the prediction column (default: 'pred').
        sub_metrics (list): List of sub-metrics to extract (e.g., ['fmeasure']).
        ref_data (list): List of data samples, each containing answer and prediction.

    Returns:
        dict: Dictionary of ROUGE scores for each metric and sub-metric.
    """
    ans, preds = [], []
    for data in ref_data:
        # index - 1 is the 'answer', last index is the prediction
        ans.append(data[1])
        preds.append(data[-1])

    # Initialize ROUGE scorer
    # egs - ['rouge1', 'rouge2', 'rougeL']
    scorer = rouge_scorer.RougeScorer(metrics, use_stemmer=True)
    rouge_rslts = {metric: {sub_metric:[] for sub_metric in sub_metrics} for metric in metrics}

    for ref, cand in tqdm(zip(ans, preds), total=len(ans)):
        scores = scorer.score(ref, cand)
        for key, data in rouge_rslts.items():
            for metric in sub_metrics:
                if metric=='fmeasure':
                    data[metric].append(scores[key].fmeasure)

    
    return rouge_rslts

def compute_bert_score(inp_data, pred_col='pred'):
    """
    Computes BERTScore precision, recall, and F1 between reference and prediction strings.

    Parameters:
        inp_data (list): List of data samples, each containing answer and prediction.
        pred_col (str): Name of the prediction column (default: 'pred').

    Returns:
        dict: Dictionary with BERTScore precision ('P'), recall ('R'), and F1 ('F1') lists.
    """
    # Extract ans & pred
    # index-1 is the 'answer', last index is the prediction
    ans = [str(data[1]) for data in inp_data]
    pred = [str(data[-1]) for data in inp_data]
    
    bert_p, bert_r, bert_f1 = bert_score.score(pred, ans, lang='en')
    
    bert_result = {'P':[], 'R':[], 'F1': []}
    for p,r,f1 in zip(bert_p, bert_r, bert_f1):
        bert_result['P'].append(p.item())
        bert_result['R'].append(r.item())
        bert_result['F1'].append(f1.item())

    return bert_result

def compute_meteor_score(inp_data, pred_col='pred'):
    """
    Calculates the METEOR score between a reference and prediction text.

    Args:
        inp_data (list): List of data samples, each containing answer and prediction.
        pred_col (str): Name of the prediction column (default: 'pred').

    Returns:
        dict: Dictionary with METEOR scores ('meteor').
    """
    ans = [str(data[1]) for data in inp_data]
    preds = [str(data[-1]) for data in inp_data]

    result = {'meteor':[]}
    for ref, cand in tqdm(zip(ans, preds), total=len(ans)):
        tokenized_reference = word_tokenize(ref)
        tokenized_hypothesis = word_tokenize(cand)
        result['meteor'].append(meteor_score([tokenized_reference], tokenized_hypothesis))
    
    return result

def compute_exact_match(inp_data, pred_col='pred'):
    """
    Computes the exact match(after lowercasing) between reference and prediction strings.

    Parameters:
        inp_data (list): List of data samples, each containing answer and prediction.
        pred_col (str): Name of the prediction column (default: 'pred').

    Returns:
        dict: Dictionary with exact match results ('exact_match'), 1 if exact match, else 0.
    """
    ans = [str(data[1]) for data in inp_data]
    preds = [str(data[-1]) for data in inp_data]

    result = {'exact_match':[]}
    for ref, cand in tqdm(zip(ans, preds), total=len(ans)):
        tokenized_reference = ref.lower() if not ref.isdigit() else ref
        tokenized_hypothesis = cand.lower() if not cand.isdigit() else cand
        result['exact_match'].append(int(tokenized_reference == tokenized_hypothesis))
    
    return result

def compute_sbert_score(inp_data):
    """
    Computes cosine similarity between sentence embeddings of reference and prediction strings using SBERT.

    Parameters:
        inp_data (list): List of data samples, each containing answer and prediction.

    Returns:
        np.ndarray: Array of cosine similarity scores for each sample.
    """
    ans = [str(data[1]) for data in inp_data]
    preds = [str(data[-1]) for data in inp_data]

    # Initialise sbert
    # Change the SBERT model here accordingly
    model = SentenceTransformer('sentence-transformers/all-roberta-large-v1')
    ans_embs = model.encode(ans)
    pred_embs = model.encode(preds)

    # Generate cosine-similarities
    sims = np.diagonal(cosine_similarity(ans_embs, pred_embs))

    return sims

def time_exec(start_time, end_time, title):
    """
    Prints the elapsed time between start_time and end_time with a custom title.

    Parameters:
        start_time (float): Start time in seconds (as returned by time.time()).
        end_time (float): End time in seconds (as returned by time.time()).
        title (str): Description of the timed operation.
    """
    elapsed_time = end_time - start_time
    print(f' > {title}: {time.strftime("%H:%M:%S", time.gmtime(elapsed_time))}')

def read_json(inp_file):
    """
    Reads a JSON or JSONL file and returns its contents.

    Parameters:
        inp_file (str): Path to the input JSON or JSONL file.

    Returns:
        object: Parsed data from the file.
    """
    if inp_file[-1] == 'l':
        # if it a .jsonl file
        with open(inp_file,'r') as f:
            inp_data = [json.loads(line) for line in f]
    else:
        inp_data = json.load(open(inp_file))

    return inp_data