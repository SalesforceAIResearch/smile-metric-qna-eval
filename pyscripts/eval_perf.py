"""
Evaluation Performance Script
Computes correlations (Pearson, Kendall's Tau, Spearman) and accuracy metrics
between SMILE and other evaluation metrics against human judgments.
"""

import json
import pandas as pd
import collections
import seaborn as sns
import pickle as pkl
import matplotlib.pyplot as plt
from rouge_score import rouge_scorer
from tqdm import tqdm
import sys
import os
sys.path.append('..')
from pyscripts.smile import SMILE
import evaluate
import random
import time
import inflect
import logging

from bert_score import score

# specific to METEOR Implementation
import nltk
from nltk.translate.meteor_score import meteor_score
from nltk.tokenize import word_tokenize
nltk.download('punkt_tab')
nltk.download('wordnet')

from scipy.stats import pearsonr, kendalltau, spearmanr
from statsmodels.stats.inter_rater import fleiss_kappa
import numpy as np
import krippendorff

def setup_logging(log_file_path='./results_logs/eval_logs/eval_perf_new.log'):
    """
    Setup logging configuration to output to both console and file.
    
    Args:
        log_file_path: Path to the log file
    """
    # Create logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Clear any existing handlers
    logger.handlers.clear()
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    simple_formatter = logging.Formatter('%(message)s')
    
    # Console handler (with simple format for readability)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(simple_formatter)
    
    # File handler (with detailed format)
    file_handler = logging.FileHandler(log_file_path, mode='a')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(detailed_formatter)
    
    # Add handlers to logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger

def get_pearson_corr(metric_a, metric_b):
    # Compute Pearson correlation
    correlation, p_value = pearsonr(metric_a, metric_b)
    
    return correlation

def compute_rouge_score(metrics:list=['rougeL'], pred_col='pred', sub_metrics=['fmeasure'], ref_data=None):
    '''
    Extracts the reference & candidate sentences, and computes the rouge score. 
    '''
    ans, preds = [], []
    for data in ref_data:
        ans.append(data['answer'])
        preds.append(data[pred_col])

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

def sort_gpt_scores(gpt4o_data, gpt3_5_data, ref_data):
    '''
    sorts the gpt_scores as per the question_id in the ref_data
    '''
    gpt4o_sorted, gpt3_5_sorted = [], []
    gpt4o_preds, gpt3_5_preds = [], []
    for data in ref_data:
        try:
            if gpt4o_data is not None:
                gpt4o_temp = gpt4o_data[str(data['question_id' if 'question_id' in data.keys() else 'id'])][0]
                if 'score' in gpt4o_temp:
                    score = gpt4o_temp['score']    
                elif 'socre' in gpt4o_temp:
                    score = gpt4o_temp['socre']
                else:
                    score = gpt4o_temp['score']
                    
                gpt4o_sorted.append(score)
                gpt4o_preds.append(gpt4o_data[str(data['question_id' if 'question_id' in data.keys() else 'id'])][0]['pred'])
            if gpt3_5_data is not None: 
                gpt3_5_temp = gpt3_5_data[str(data['question_id' if 'question_id' in data.keys() else 'id'])][0]
                if 'score' in gpt3_5_temp:
                    score = gpt3_5_temp['score']    
                elif 'socre' in gpt3_5_temp:
                    score = gpt3_5_temp['socre']
                else:
                    score = gpt3_5_temp['score']

                if 'pred' in gpt3_5_temp:
                    pred = gpt3_5_temp['pred']
                elif 'pre dasdf' in gpt3_5_temp:
                    pred = gpt3_5_temp['pre dasdf']
                else:
                    pred = gpt3_5_temp['pred']
                    
                gpt3_5_sorted.append(score)
                gpt3_5_preds.append(pred)
        except Exception as e:
            logging.error(f'Error processing question_id: {e}')
            if gpt4o_data is not None:
                logging.debug(f"GPT-4o data: {gpt4o_data[str(data['question_id' if 'question_id' in data.keys() else 'id'])][0]}")
                gpt4o_sorted.append(-1)
            if gpt3_5_data is not None:
                logging.debug(f"GPT-3.5 data: {gpt3_5_data[str(data['question_id' if 'question_id' in data.keys() else 'id'])][0]}")
                gpt3_5_sorted.append(-1)
            
    return gpt4o_sorted, gpt3_5_sorted, gpt4o_preds, gpt3_5_preds

def get_kendalltau(metric_a, metric_b):
    '''
    Computes kendall's tau (default is tau-b)
    '''
    # Calculate Kendall's Tau
    tau, p_value = kendalltau(metric_a, metric_b)
    return tau, p_value
    
def cal_acc(vals, gpt_eval=False, threshold_eval=False, exact_eval=False, threshold=0.5, verbose=False, title=None):
    '''
    Computes accuracy
     > theshold : <>
    '''
    if gpt_eval:
        yes_count = no_count = invalid_cnt = 0
        for val in vals:
            try:
                # Computing accuracy
                if "yes" in val.lower():
                    yes_count += 1
                elif "no" in val.lower():
                    no_count += 1
                else:
                    invalid_cnt += 1
            except:
                print(accuracy)
                
        accuracy = yes_count / (yes_count + no_count + invalid_cnt)
        mean_val = accuracy
        if verbose: print(f' > Yes, No, Invalid: ({yes_count}, {no_count}, {invalid_cnt})')

    elif threshold_eval:
        accuracy = (np.array(vals)>=threshold).sum()/len(vals)
        mean_val = np.mean(vals)

    elif exact_eval:
        print(len(vals))
        accuracy = (np.array(vals)==threshold).sum()/len(vals)
        mean_val = np.mean(vals)

    if verbose: print(f' > {title} (accuracy): {accuracy:.3f} (avg val: {mean_val:.3f})')

    return accuracy
    
def get_bins(eval_scores, num_bins=6, bin_cat='numpy'):
    if bin_cat == 'numpy':
        # Define min and max values
        min_val = 0
        max_val = 1
    
        # Calculate bin edges (6 bins means we need 7 edges)
        bin_edges = np.linspace(min_val, max_val, num_bins+1)
    
        # Assign each value to a bin
        bin_indices = np.digitize(eval_scores, bin_edges) - 1
    
        # Ensure the values are within the range
        bin_indices = np.clip(bin_indices, 0, num_bins-1)
    elif bin_cat == 'lave':
        classes = np.where(values < 1/3, 1, np.where(values < 2/3, 2, 3))
        bin_indices = (classes-1)/2


    return bin_indices
    
def smile_wt_exp(smile_scores, human_scores=None, gpt_scores=None, use_bins=False, num_bins=6, use_index=None):
    '''
    Computes smile score against human_scores/ gpt_scores and returns pearson correlation
    Returns dictionary with weight experiments results
    '''
    if use_index is None: use_index=slice(None)
    wts = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    
    # extract scores
    sent_scores = smile_scores['sent_emb_scores']
    kwd_scores = smile_scores['kwd_scores'] # contains both keyword embedding scores and fraction match scores

    # Store results
    wt_results = {}
    
    logging.info('Weighted-SMILE experiments')
    logging.info(f"Average SMILE score: {smile_scores['avg'].mean()}")
    if gpt_scores is not None:
        gpt_pc = get_pearson_corr(gpt_scores[use_index], human_scores[use_index])
        gpt_kt,_ = get_kendalltau(gpt_scores[use_index], human_scores[use_index])
        spc, _ = spearmanr(gpt_scores[use_index], human_scores[use_index])
        logging.info(f"PC (GPT, human_score): {gpt_pc:.4f}")
        logging.info(f"KT (GPT, human_score): {gpt_kt:.4f}")
        logging.info(f"SP (GPT, human_score): {spc:.4f}")
    
    for wt in wts:
        wt_smile_scores = wt*sent_scores + (1-wt)*kwd_scores

        if use_bins: wt_smile_scores_bins = get_bins(wt_smile_scores, num_bins=num_bins)
        
        wt_results[wt] = {}
            
        # Check for pearson correlation
        if human_scores is not None:
            pc = get_pearson_corr(wt_smile_scores[use_index], human_scores[use_index])
            kt,_ = get_kendalltau(wt_smile_scores_bins[use_index], human_scores[use_index])
            spc, _ = spearmanr(wt_smile_scores[use_index], human_scores[use_index])
            acc = cal_acc(wt_smile_scores_bins, threshold_eval=True, threshold=4, verbose=False, title='smile_avg')
            
            wt_results[wt]['pearson'] = pc
            wt_results[wt]['kendall'] = kt
            wt_results[wt]['spearman'] = spc
            wt_results[wt]['accuracy'] = acc
            
            logging.info(f"Wt PC ({wt}, human_score): {pc:.4f} | Wt KT ({wt}, human_score): {kt:.4f} | Wt SPC ({wt}, human_score): {spc:.4f}")
            
        if gpt_scores is not None:
            pc = get_pearson_corr(wt_smile_scores, gpt_scores)
            acc = cal_acc(wt_smile_scores_bins, threshold_eval=True, threshold=4, verbose=False, title='smile_avg')
            wt_results[wt]['pearson_gpt'] = pc
            wt_results[wt]['accuracy_gpt'] = acc
            logging.info(f"Wt PC ({wt}, gpt_scores): {pc:.4f}")

    return wt_results 

def convert_scores(scores):
    return (scores+1)/2
    
def new_cosine_values(smile_scores):
    '''
    converts smile in the range [0,1]
    '''
    sent_scores = smile_scores['sent_emb_scores']
    kwd_emb_scores = smile_scores['kwd_emb_scores']
    frac_exact_scores = smile_scores['frac_exact_match']
    
    new_sent_scores = convert_scores(sent_scores)
    new_kwd_emb_scores = convert_scores(kwd_emb_scores)
    new_kwd_scores = (new_kwd_emb_scores + frac_exact_scores)/2
    new_avg = (new_sent_scores + new_kwd_scores)/2
    new_scores = get_bins(new_avg)

    # re-assign the values
    smile_scores['sent_emb_scores'] = new_sent_scores
    smile_scores['kwd_emb_scores'] = new_kwd_emb_scores
    smile_scores['kwd_scores'] = new_kwd_scores
    smile_scores['avg'] = new_avg
    smile_scores['avg score'] = new_scores

    return smile_scores
    
def get_easy_match(data, answer_key='answer'):
    """
    Calculate easy match scores.
    
    Args:
        data: List of dictionaries containing predictions and answers
        answer_key: Key to use for the answer field ('answer' or 'syn_ans')
    
    Returns:
        numpy array of scores (0 or 1)
    """
    scores = []
    for val in data:
        answer = val[answer_key].lower()
        pred = val['pred'].lower()
        if pred in answer or answer in pred:
            scores.append(1)
        else:
            scores.append(0)

    return np.array(scores)

def load_eval_data(ref_dataset, data_paths, rerange_smile_vals=False, use_human_scores=False, split_qa_types:str=None, use_synans=False):
    """
    Load evaluation data from various sources.
    
    Args:
        ref_dataset: Reference dataset
        data_paths: Dictionary of paths to data files
        rerange_smile_vals: Whether to rerange SMILE values
        use_human_scores: Whether to use human scores
        split_qa_types: Type of QA split
        use_synans: Whether to use synthetic answers (affects easy_match key)
    
    Returns:
        Dictionary containing loaded scores
    """
    smile_score = pkl.load(open(data_paths['smile_score'], 'rb'))
    gpt4o_score = json.load(open(data_paths['gpt4o_score']))
    gpt3_5_score = json.load(open(data_paths['gpt3.5_score']))
    bert_score = pkl.load(open(data_paths['bert_score'], 'rb'))
    rouge_score = pkl.load(open(data_paths['rouge_score'], 'rb'))
    meteor_score = pkl.load(open(data_paths['meteor_score'], 'rb'))['meteor']
    exact_match_score = pkl.load(open(data_paths['exact_match'], 'rb'))['exact_match']
    sbert_score = pkl.load(open(data_paths['sbert_score'], 'rb'))
    
    # Load BLEURT and MoverScore if available in data_paths
    bleurt_score = pkl.load(open(data_paths['bleurt_score'], 'rb'))['scores'] if 'bleurt_score' in data_paths else None
    moverscore_score = pkl.load(open(data_paths['moverscore_score'], 'rb'))['scores'] if 'moverscore_score' in data_paths else None
    print(len(bleurt_score), len(moverscore_score))

    if rerange_smile_vals:
        smile_score = new_cosine_values(smile_score)
    
    # Sort the gpt-scores as per the qids
    gpt4o_sorted, gpt3_5_sorted, gpt4o_preds, gpt3_5_preds= sort_gpt_scores(gpt4o_data = gpt4o_score,
                                                  gpt3_5_data = gpt3_5_score,
                                                  ref_data = ref_dataset)

    # Use syn_ans key when use_synans is True, otherwise use answer key
    answer_key = 'syn_ans' if use_synans else 'answer'
    easy_scores = get_easy_match(ref_dataset, answer_key=answer_key)
    logging.info(f"Easy match using answer key: '{answer_key}'")

    human_scores=[]
    if use_human_scores:
        smile_score['avg'] = smile_score['avg'][:25]
        smile_score['hm'] = smile_score['hm'][:25]
        smile_score['avg score'] = smile_score['avg score'][:25]
        smile_score['hm score'] = smile_score['hm score'][:25]
        smile_score['sent_emb_scores'] = smile_score['sent_emb_scores'][:25]
        smile_score['kwd_emb_scores'] = smile_score['kwd_emb_scores'][:25]
        smile_score['kwd_scores'] = smile_score['kwd_scores'][:25]
        rouge_score['rougeL']['fmeasure'] = rouge_score['rougeL']['fmeasure'][:25]
        bert_score['F1'] = bert_score['F1'][:25]
        meteor_score = meteor_score[:25]
        exact_match_score = exact_match_score[:25]
        easy_scores = easy_scores[:25]
        sbert_score = sbert_score[:25]
        if bleurt_score is not None:
            bleurt_score = bleurt_score[:25]
        if moverscore_score is not None:
            moverscore_score = moverscore_score[:25]
        human_scores = [data['human_rating'] for data in ref_dataset]

    if split_qa_types is not None:
        sent_thresh = 11
        valid_indices = [i for i, data in enumerate(ref_dataset) if len(data['pred'].strip().split())<=sent_thresh] if split_qa_types=='short' else [i for i, data in enumerate(ref_dataset) if len(data['pred'].strip().split())>sent_thresh]
        # Extract all the scores accordingly - 
        smile_score['avg'] = smile_score['avg'][valid_indices]
        smile_score['hm'] = smile_score['hm'][valid_indices]
        smile_score['avg score'] = smile_score['avg score'][valid_indices]
        smile_score['hm score'] = smile_score['hm score'][valid_indices]
        smile_score['sent_emb_scores'] = smile_score['sent_emb_scores'][valid_indices]
        smile_score['kwd_emb_scores'] = smile_score['kwd_emb_scores'][valid_indices]
        smile_score['kwd_scores'] = smile_score['kwd_scores'][valid_indices]
        rouge_score['rougeL']['fmeasure'] = np.array(rouge_score['rougeL']['fmeasure'])[valid_indices]
        bert_score['F1'] = np.array(bert_score['F1'])[valid_indices]
        meteor_score = np.array(meteor_score)[valid_indices]
        exact_match_score = np.array(exact_match_score)[valid_indices]
        easy_scores = np.array(easy_scores)[valid_indices]
        sbert_score = np.array(sbert_score)[valid_indices]
        if bleurt_score is not None:
            bleurt_score = np.array(bleurt_score)[valid_indices]
        if moverscore_score is not None:
            moverscore_score = np.array(moverscore_score)[valid_indices]
        human_scores = np.array(human_scores)[valid_indices]
        gpt4o_sorted, gpt3_5_sorted = np.array(gpt4o_sorted)[valid_indices], np.array(gpt3_5_sorted)[valid_indices]

    print(len(bleurt_score), len(moverscore_score))

    result = {
        'smile_score' : smile_score,
        'rouge_score' : rouge_score,
        'bert_score' : bert_score,
        'meteor_score' : meteor_score,
        'exact_match_score' : exact_match_score,
        'easy_score' : easy_scores,
        'sbert_score' : sbert_score,
        'gpt4o_sorted': gpt4o_sorted,
        'gpt3_5_sorted' : gpt3_5_sorted,
        'gpt4o_preds': gpt4o_preds,
        'gpt3_5_preds': gpt3_5_preds,
        'gpt4o_score': gpt4o_score,
        'gpt3_5_score': gpt3_5_score
    }
    if bleurt_score is not None:
        result['bleurt_score'] = bleurt_score
    if moverscore_score is not None:
        result['moverscore_score'] = moverscore_score
    if use_human_scores: result['human_score'] = human_scores

    return result

def merge_domain_data(domain_data:dict=None, ref_data:list[dict]=None, params:dict=None):
    required_keys = ['ref_data','smile_score', 'rouge_score', 'bert_score', 'meteor_score', 'exact_match_score', 'easy_score', 'sbert_score', 'gpt4o_sorted', 'gpt3_5_sorted', 'gpt4o_preds', 'gpt3_5_preds', 'gpt4o_score', 'gpt3_5_score']
    if not domain_data:
        pass
    
    # Merge each metric value
    for key in required_keys:
        key_data = domain_data.get(key, None)
        if key_data is None:
            if key == 'ref_data':
                domain_data[key] = ref_data
            else:
                domain_data[key] = params[key]
        else:
            if key == 'ref_data':
                domain_data[key] = key_data + ref_data
            elif key == 'gpt4o_score' or key == 'gpt3_5_score':
                domain_data[key] = key_data | params[key]
            
            else:
                if key == 'rouge_score':
                    if isinstance(domain_data[key]['rougeL']['fmeasure'], list):
                        domain_data[key]['rougeL']['fmeasure'] = domain_data[key]['rougeL']['fmeasure'] + params[key]['rougeL']['fmeasure']
                    else:
                        domain_data[key]['rougeL']['fmeasure'] = np.concatenate((domain_data[key]['rougeL']['fmeasure'], params[key]['rougeL']['fmeasure']))
                elif isinstance(key_data, dict):
                    for k,v in key_data.items():
                        if isinstance(v, list):
                            domain_data[key][k] = v + params[key][k]
                        elif isinstance(v, np.ndarray):
                            domain_data[key][k] = np.concatenate((v, params[key][k]))
                        elif isinstance(v, dict):
                            raise ValueError(f'{key}->{k} still has dictionary data')
                elif isinstance(key_data, list):
                    domain_data[key] = key_data + params[key]
                elif isinstance(key_data, np.ndarray):
                    domain_data[key] = np.concatenate((key_data, params[key]))
                    
    if 'human_score' in params.keys():
        key_data = domain_data.get('human_score', None)
        if key_data is None:
            domain_data['human_score'] = params['human_score']
        else:
            if isinstance(key_data, list):
                domain_data['human_score'] = key_data + params['human_score']
            elif isinstance(key_data, np.ndarray):
                domain_data['human_score'] = np.concatenate((key_data, params['human_score']))
    
    # Handle bleurt_score and moverscore_score if present
    for optional_key in ['bleurt_score', 'moverscore_score']:
        if optional_key in params.keys() and params[optional_key] is not None:
            key_data = domain_data.get(optional_key, None)
            if key_data is None:
                domain_data[optional_key] = params[optional_key]
            else:
                if isinstance(key_data, list):
                    domain_data[optional_key] = key_data + params[optional_key]
                elif isinstance(key_data, np.ndarray):
                    domain_data[optional_key] = np.concatenate((key_data, params[optional_key]))
    
    return domain_data
    
def eval_data(ref_dataset, dataset_name, model_name, pred_col='pred', eval_metrics=['pearson','kendall-tau'], data_paths=None, use_human_scores=False, smile_wt_exps=False, get_acc=False, rerange_smile_vals=False, split_qa_types:str=None, use_synans=False, **kwargs):
    '''
    Runs pearson and rouge evaluation on provided 'dataset' (utilises 'smile', 'gpt-4o' & 'gpt-3.5-turbo' scores)
    '''
    if not data_paths:
        logging.error('Please provide valid data paths')
        return
        
    # Open the scoring files 
    logging.info('1. Loading the scores')
    if not kwargs:
        # load the data
        params = load_eval_data(ref_dataset = ref_dataset,
                                data_paths = data_paths,
                                rerange_smile_vals = rerange_smile_vals,
                                use_human_scores = use_human_scores,
                                split_qa_types = split_qa_types,
                                use_synans = use_synans
                               )
    else:
        # if kwargs are given directly load all the scores
        required_keys = ['smile_score', 'rouge_score', 'bert_score', 'meteor_score', 'exact_match_score', 'easy_score', 'sbert_score', 'gpt4o_sorted', 'gpt3_5_sorted', 'gpt4o_preds', 'gpt3_5_preds', 'gpt4o_score', 'gpt3_5_score', 'bleurt_score','moverscore_score']
        params = {}
        for key in required_keys:
            if key not in kwargs:
                raise KeyError(f"Missing required keyword args: '{key}'")
            params[key] = kwargs[key]
        if use_human_scores:
            if 'human_score' not in kwargs:
                raise KeyError(f"Missing required keyword args: 'human_score'")
            params['human_score'] = kwargs['human_score']
            
    # extract all the values
    if use_human_scores: human_scores = params['human_score']
    smile_score = params['smile_score']
    rouge_score = params['rouge_score']
    bert_score = params['bert_score']
    meteor_score = params['meteor_score']
    exact_match_score = params['exact_match_score']
    easy_scores = params['easy_score']
    sbert_score = params['sbert_score']
    gpt4o_sorted, gpt3_5_sorted = params['gpt4o_sorted'], params['gpt3_5_sorted']
    gpt4o_preds, gpt3_5_preds = params['gpt4o_preds'], params['gpt3_5_preds']
    gpt4o_score, gpt3_5_score = params['gpt4o_score'], params['gpt3_5_score']
    bleurt_score = params['bleurt_score']
    moverscore_score = params['moverscore_score']
            

    logging.info(f"smile_score: {len(smile_score['avg score'])}, gpt4o_sorted: {len(gpt4o_sorted)}")
    logging.info(f"avg avg: {smile_score['avg'].mean():.3f}, avg hm: {smile_score['hm'].mean():.3f}")
    logging.info(f'{dataset_name} (Accuracy):')
    
    # Initialize accuracy variables
    human_acc = None
    gpt4o_acc = None
    gpt3_5_acc = None
    smile_avg_acc = None
    rouge_acc = None
    bert_acc = None
    meteor_acc = None
    exact_match_acc = None
    sbert_acc = None
    bleurt_acc = None
    moverscore_acc = None
    easy_match_acc = None
    
    if get_acc:
        # Calculate human accuracy
        if use_human_scores: 
            human_acc = cal_acc(human_scores, exact_eval=True, threshold=2, verbose=True, title='human-exact')
        gpt4o_acc = cal_acc(gpt4o_preds, gpt_eval=True, verbose=True, title='gpt4o')
        gpt3_5_acc = cal_acc(gpt3_5_preds, gpt_eval=True, verbose=True, title='gpt3_5')
        smile_avg_acc = cal_acc(smile_score['avg score'], threshold_eval=True, threshold=4, verbose=True, title='smile_avg_bin')
        rouge_acc = cal_acc(rouge_score['rougeL']['fmeasure'], threshold_eval=True, threshold=0.5, verbose=True, title='rouge_acc')
        bert_acc = cal_acc(bert_score['F1'], threshold_eval=True, threshold=0.5, verbose=True, title='bert_acc')
        meteor_acc = cal_acc(meteor_score, threshold_eval=True, threshold=0.5, verbose=True, title='meteor_acc')
        exact_match_acc = cal_acc(exact_match_score, exact_eval=True, threshold=1, verbose=True, title='exact_match_acc')
        sbert_acc = cal_acc(sbert_score, threshold_eval=True, threshold=0.5, verbose=True, title='sbert_acc')
        if bleurt_score is not None:
            bleurt_acc = cal_acc(bleurt_score, threshold_eval=True, threshold=0.5, verbose=True, title='bleurt_acc')
        if moverscore_score is not None:
            moverscore_acc = cal_acc(moverscore_score, threshold_eval=True, threshold=0.5, verbose=True, title='moverscore_acc')
        easy_match_acc = cal_acc(easy_scores, exact_eval=True, threshold=1, verbose=True, title='easy_match_acc')

    # Initialize weight experiments results
    weight_exp_results = None
    
    if smile_wt_exps:
        weight_exp_results = smile_wt_exp(smile_score, human_scores, gpt4o_sorted, use_bins=True)

    # extract pearson-correlation (pc) for each pair
    if split_qa_types is not None: logging.info(f'QA Type - {split_qa_types}')
    logging.info('3. Extracting relevant metrics')
    smile_metric='avg score'
    if 'pearson' in eval_metrics:
        logging.info(' > Computing Pearson Correlation')

        if use_human_scores:
            # Using GPT
            pc_human_gpt4o = get_pearson_corr(human_scores, gpt4o_sorted)
            pc_human_gpt3_5 = get_pearson_corr(human_scores, gpt3_5_sorted)
            pc_human_smile_avg = get_pearson_corr(human_scores, smile_score['avg'])
            pc_human_smile_hm = get_pearson_corr(human_scores, smile_score['hm score'])
            pc_human_smile_sent = get_pearson_corr(human_scores, smile_score['sent_emb_scores'])
            pc_human_smile_kwd = get_pearson_corr(human_scores, smile_score['kwd_emb_scores'])
            pc_human_rouge = get_pearson_corr(human_scores, rouge_score['rougeL']['fmeasure'])
            pc_human_bert = get_pearson_corr(human_scores, bert_score['F1'])
            pc_human_meteor = get_pearson_corr(human_scores, meteor_score)
            pc_human_em = get_pearson_corr(human_scores, exact_match_score)
            pc_human_sbert = get_pearson_corr(human_scores, sbert_score)
            pc_human_bleurt = None
            pc_human_moverscore = None
            if bleurt_score is not None:
                pc_human_bleurt = get_pearson_corr(human_scores, bleurt_score)
            if moverscore_score is not None:
                pc_human_moverscore = get_pearson_corr(human_scores, moverscore_score)
            pc_human_easy = get_pearson_corr(human_scores, easy_scores)

        else:
            # Using GPT
            pc_gpt4o_smile = get_pearson_corr(gpt4o_sorted, smile_score['avg'])
            pc_gpt3_5_smile = get_pearson_corr(gpt3_5_sorted, smile_score['avg'])
            pc_gpt4o_gpt3_5 = get_pearson_corr(gpt4o_sorted, gpt3_5_sorted)
    
            # Using ROUGE
            pc_gpt4o_rougeL = get_pearson_corr(gpt4o_sorted, rouge_score['rougeL']['fmeasure'])
            pc_smile_rougeL = get_pearson_corr(smile_score['avg score'], rouge_score['rougeL']['fmeasure'])
    
            # Using BERTScore
            pc_gpt4o_bert = get_pearson_corr(gpt4o_sorted, bert_score['F1'])
            pc_smile_bert = get_pearson_corr(smile_score['avg score'], bert_score['F1'])

            # USING meteor
            pc_gpt4o_meteor = get_pearson_corr(gpt4o_sorted, meteor_score)

            # USING exact match
            pc_gpt4o_em = get_pearson_corr(gpt4o_sorted, exact_match_score)

            # USING sbert
            pc_gpt4o_sbert = get_pearson_corr(gpt4o_sorted, sbert_score)

            pc_gpt4o_easy = get_pearson_corr(gpt4o_sorted, easy_scores)
    
    if 'kendall-tau' in eval_metrics:
        logging.info(" > Computing Kendall's tau(b)")

        if use_human_scores:
            # Using GPT
            kt_human_gpt4o, _ = get_kendalltau(human_scores, gpt4o_sorted)
            kt_human_gpt3_5, _ = get_kendalltau(human_scores, gpt3_5_sorted)
            kt_human_smile_avg, _ = get_kendalltau(human_scores, smile_score['avg score'])
            kt_human_smile_hm, _ = get_kendalltau(human_scores, smile_score['hm score'])
            kt_human_smile_sent, _ = get_kendalltau(human_scores, smile_score['sent_emb_scores'])
            kt_human_smile_kwd, _ = get_kendalltau(human_scores, smile_score['kwd_emb_scores'])
            kt_human_rouge, _ = get_kendalltau(human_scores, rouge_score['rougeL']['fmeasure'])
            kt_human_bert, _ = get_kendalltau(human_scores, bert_score['F1'])
            kt_human_meteor, _ = get_kendalltau(human_scores, meteor_score)
            kt_human_em, _ = get_kendalltau(human_scores, exact_match_score)
            kt_human_sbert, _ = get_kendalltau(human_scores, sbert_score)
            kt_human_bleurt = None
            kt_human_moverscore = None
            if bleurt_score is not None:
                kt_human_bleurt, _ = get_kendalltau(human_scores, bleurt_score)
            if moverscore_score is not None:
                kt_human_moverscore, _ = get_kendalltau(human_scores, moverscore_score)
            kt_human_easy, _ = get_kendalltau(human_scores, easy_scores)

        else:
            # Using GPT
            kt_gpt4o_smile, kt_gpt4o_smile_pval = get_kendalltau(gpt4o_sorted, smile_score['avg'])
            kt_gpt3_5_smile, kt_gpt3_5_smile_pval  = get_kendalltau(gpt3_5_sorted, smile_score['avg'])
            kt_gpt4o_gpt3_5, kt_gpt4o_gpt3_5_pval = get_kendalltau(gpt4o_sorted, gpt3_5_sorted)
    
            # Using ROUGE
            kt_gpt4o_rougeL, kt_gpt4o_rougeL_pval = get_kendalltau(gpt4o_sorted, rouge_score['rougeL']['fmeasure'])
            kt_smile_rougeL, kt_smile_rougeL_pval = get_kendalltau(smile_score['avg score'], rouge_score['rougeL']['fmeasure'])
    
            # Using BERTScore
            kt_gpt4o_bert, kt_gpt4o_bert_pval = get_kendalltau(gpt4o_sorted, bert_score['F1'])
            kt_smile_bert, kt_smile_bert_pval = get_kendalltau(smile_score['avg score'], bert_score['F1'])

            # Using METEOR
            kt_gpt4o_meteor, _ = get_kendalltau(gpt4o_sorted, meteor_score)

            # Using Exact Match
            kt_gpt4o_em, _ = get_kendalltau(gpt4o_sorted, exact_match_score)

            # Using sBERT
            kt_gpt4o_sbert, _ = get_kendalltau(gpt4o_sorted, sbert_score)
            kt_gpt4o_easy, _ = get_kendalltau(gpt4o_sorted, easy_scores)

    # print the analysis
    logging.info('4. Analysis')
    if 'pearson' in eval_metrics:
        logging.info(f' Pearson Correlation ({dataset_name}) -')
        if use_human_scores:
            output_lines = [
                f"  1. human & gpt-4o : {pc_human_gpt4o:.3f}",
                f"  2. human & gpt-3.5 : {pc_human_gpt3_5:.3f}",
                f"  3. human & smile (avg) : {pc_human_smile_avg:.3f}",
                f"  4. human & smile (hm) : {pc_human_smile_hm:.3f}",
                f"  5. human & rouge : {pc_human_rouge:.3f}",
                f"  6. human & bert : {pc_human_bert:.3f}",
                f"  7. human & smile (sent) : {pc_human_smile_sent:.3f}",
                f"  8. human & smile (kwd) : {pc_human_smile_kwd:.3f}",
                f"  9. human & meteor: {pc_human_meteor:.3f}",
                f" 10. human & exact match: {pc_human_em:.3f}",
                f" 11. human & sbert: {pc_human_sbert:.3f}",
            ]
            if bleurt_score is not None:
                output_lines.append(f" 12. human & bleurt: {pc_human_bleurt:.3f}")
            if moverscore_score is not None:
                output_lines.append(f" 13. human & moverscore: {pc_human_moverscore:.3f}")
            output_lines.append(f" 14. human & easy match: {pc_human_easy:.3f}")
            logging.info('\n'.join(output_lines))
        else:
            logging.info(f"  1. gpt-4o & smile : {pc_gpt4o_smile:.3f}\n  2. gpt-3.5-turbo & smile : {pc_gpt3_5_smile:.3f}\n  3. gpt-4o & gpt-3.5-turbo : {pc_gpt4o_gpt3_5:.3f}")
            logging.info("\n---------")
            logging.info(f' Pearson Correlation ({dataset_name}, using ROUGE)-\n  1. gpt4o & rougeL : {pc_gpt4o_rougeL:.3f}\n  2. smile & rougeL : {pc_smile_rougeL:.3f}')
    
    if 'kendall-tau' in eval_metrics:
        logging.info(f" Kendall's Tau ({dataset_name}) -")
        if use_human_scores:
            output_lines = [
                f"  1. human & gpt-4o : {kt_human_gpt4o:.3f}",
                f"  2. human & gpt-3.5 : {kt_human_gpt3_5:.3f}",
                f"  3. human & smile (avg) : {kt_human_smile_avg:.3f}",
                f"  4. human & smile (hm) : {kt_human_smile_hm:.3f}",
                f"  5. human & rouge : {kt_human_rouge:.3f}",
                f"  6. human & bert : {kt_human_bert:.3f}",
                f"  7. human & smile (sent) : {kt_human_smile_sent:.3f}",
                f"  8. human & smile (kwd) : {kt_human_smile_kwd:.3f}",
                f"  9. human & meteor: {kt_human_meteor:.3f}",
                f" 10. human & exact match: {kt_human_em:.3f}",
                f" 11. human & sbert: {kt_human_sbert:.3f}",
            ]
            if bleurt_score is not None:
                output_lines.append(f" 12. human & bleurt: {kt_human_bleurt:.3f}")
            if moverscore_score is not None:
                output_lines.append(f" 13. human & moverscore: {kt_human_moverscore:.3f}")
            output_lines.append(f" 14. human & easy match: {kt_human_easy:.3f}")
            logging.info('\n'.join(output_lines))


    result = {
        'smile_scores': smile_score, 
        'gpt4o_scores': gpt4o_sorted, 
        'gpt3.5_scores': gpt3_5_sorted, 
        'rouge_scores': rouge_score, 
        'bert_scores': bert_score, 
        'human_scores': human_scores, 
        'gpt4o_scores_raw': gpt4o_score, 
        'sbert_score': sbert_score, 
        'exact_match': exact_match_score, 
        'easy_match': easy_scores
    }
    if bleurt_score is not None:
        result['bleurt_score'] = bleurt_score
    if moverscore_score is not None:
        result['moverscore_score'] = moverscore_score
    
    # Add correlation results if calculated
    if use_human_scores and 'pearson' in eval_metrics:
        result['pearson_correlations'] = {
            'Exact Match': pc_human_em,
            'Easy Match': pc_human_easy,
            'ROUGE-L': pc_human_rouge,
            'METEOR': pc_human_meteor,
            'BERTScore': pc_human_bert,
            'sBERT': pc_human_sbert,
            'BLEURT': pc_human_bleurt,
            'Moverscore': pc_human_moverscore,
            'GPT-3.5': pc_human_gpt3_5,
            'GPT-4o': pc_human_gpt4o,
            'SMILE': pc_human_smile_avg
        }
    
    if use_human_scores and 'kendall-tau' in eval_metrics:
        result['kendall_correlations'] = {
            'Exact Match': kt_human_em,
            'Easy Match': kt_human_easy,
            'ROUGE-L': kt_human_rouge,
            'METEOR': kt_human_meteor,
            'BERTScore': kt_human_bert,
            'sBERT': kt_human_sbert,
            'BLEURT': kt_human_bleurt,
            'Moverscore': kt_human_moverscore,
            'GPT-3.5': kt_human_gpt3_5,
            'GPT-4o': kt_human_gpt4o,
            'SMILE': kt_human_smile_avg
        }
    
    # Add accuracy results if calculated
    if get_acc:
        result['accuracies'] = {
            'Human': human_acc if human_acc is not None else np.nan,
            'Exact Match': exact_match_acc if exact_match_acc is not None else np.nan,
            'Easy Match': easy_match_acc if easy_match_acc is not None else np.nan,
            'ROUGE-L': rouge_acc if rouge_acc is not None else np.nan,
            'METEOR': meteor_acc if meteor_acc is not None else np.nan,
            'BERTScore': bert_acc if bert_acc is not None else np.nan,
            'sBERT': sbert_acc if sbert_acc is not None else np.nan,
            'BLEURT': bleurt_acc if bleurt_acc is not None else np.nan,
            'Moverscore': moverscore_acc if moverscore_acc is not None else np.nan,
            'GPT-3.5': gpt3_5_acc if gpt3_5_acc is not None else np.nan,
            'GPT-4o': gpt4o_acc if gpt4o_acc is not None else np.nan,
            'SMILE': smile_avg_acc if smile_avg_acc is not None else np.nan
        }
    
    # Add weight experiment results if calculated
    if weight_exp_results is not None:
        result['weight_experiments'] = weight_exp_results
    
    return result

def create_and_export_correlation_tables(pearson_correlations, kendall_correlations, output_path='./results_logs/result_tables/', print_tables=True):
    """
    Create pandas DataFrames from correlation dictionaries and export to CSV.
    
    Args:
        pearson_correlations: Dictionary with dataset names as keys and Pearson correlation dicts as values
        kendall_correlations: Dictionary with dataset names as keys and Kendall correlation dicts as values
        output_path: Directory path where output files will be saved
        print_tables: Whether to print the tables to console/log (default: False)
    
    Returns:
        tuple: (pearson_df, kendall_df, csv_pearson_filename, csv_kendall_filename)
    """
    logging.info("\n" + "="*80)
    logging.info("Creating Correlation Tables...")
    logging.info("="*80 + "\n")
    
    # Define the order of datasets as per the image
    dataset_order = ['TGIF', 'MSVD', 'MSRVTT', 'TextVQA', 'DocVQA', 'POPE', 
                     'MRQA', 'HotpotQA', 'MuSiQue']
    
    # Define the order of metrics
    metric_order = ['Exact Match', 'Easy Match', 'ROUGE-L', 'METEOR', 'BERTScore', 
                    'sBERT', 'BLEURT', 'Moverscore', 'GPT-3.5', 'GPT-4o', 'SMILE']
    
    # Create Pearson correlation DataFrame
    pearson_df = pd.DataFrame(pearson_correlations)
    available_datasets = [d for d in dataset_order if d in pearson_df.columns]
    pearson_df = pearson_df[available_datasets]
    pearson_df = pearson_df.reindex(metric_order)
    
    # Create Kendall correlation DataFrame
    kendall_df = pd.DataFrame(kendall_correlations)
    available_datasets = [d for d in dataset_order if d in kendall_df.columns]
    kendall_df = kendall_df[available_datasets]
    kendall_df = kendall_df.reindex(metric_order)
    
    # Display the tables with nice formatting (if requested)
    if print_tables:
        logging.info("\n" + "="*120)
        logging.info("Pearson Correlation")
        logging.info("="*120)
        logging.info(pearson_df.to_string(float_format=lambda x: f'{x:.3f}' if pd.notna(x) else 'nan'))
        logging.info("\n")
        
        logging.info("\n" + "="*120)
        logging.info("Kendall's Tau")
        logging.info("="*120)
        logging.info(kendall_df.to_string(float_format=lambda x: f'{x:.3f}' if pd.notna(x) else 'nan'))
        logging.info("\n")
    
    # Save to CSV
    csv_pearson_filename = f'{output_path}pearson_correlation.csv'
    csv_kendall_filename = f'{output_path}kendall_correlation.csv'
    
    # Save to CSV with 3 decimal places
    pearson_df.to_csv(csv_pearson_filename, float_format='%.3f')
    kendall_df.to_csv(csv_kendall_filename, float_format='%.3f')
    
    logging.info(f"\n{'='*80}")
    logging.info(f"Results saved to:")
    logging.info(f"  CSV (Pearson): {csv_pearson_filename}")
    logging.info(f"  CSV (Kendall): {csv_kendall_filename}")
    logging.info(f"{'='*80}\n")
    
    return pearson_df, kendall_df, csv_pearson_filename, csv_kendall_filename

def create_and_export_accuracy_table(accuracies, output_path='./results_logs/result_tables/', print_table=True):
    """
    Create pandas DataFrame from accuracy dictionary and export to CSV.
    """
    logging.info("\n" + "="*80)
    logging.info("Creating Accuracy Table...")
    logging.info("="*80 + "\n")
    
    dataset_order = ['TGIF', 'MSVD', 'MSRVTT', 'TextVQA', 'DocVQA', 'POPE', 
                     'MRQA', 'HotpotQA', 'MuSiQue']
    
    metric_order = ['Human', 'Exact Match', 'Easy Match', 'ROUGE-L', 'METEOR', 'BERTScore', 
                    'sBERT', 'BLEURT', 'Moverscore', 'GPT-3.5', 'GPT-4o', 'SMILE']
    
    # Create Accuracy DataFrame
    accuracy_df = pd.DataFrame(accuracies)
    available_datasets = [d for d in dataset_order if d in accuracy_df.columns]
    accuracy_df = accuracy_df[available_datasets]
    accuracy_df = accuracy_df.reindex(metric_order)
    
    if print_table:
        logging.info("\n" + "="*120)
        logging.info("Accuracy")
        logging.info("="*120)
        logging.info(accuracy_df.to_string(float_format=lambda x: f'{x:.3f}' if pd.notna(x) else 'nan'))
        logging.info("\n")
    
    # Save to CSV
    csv_filename = f'{output_path}accuracy.csv'
    accuracy_df.to_csv(csv_filename, float_format='%.3f')
    
    logging.info(f"\n{'='*80}")
    logging.info(f"Accuracy results saved to: {csv_filename}")
    logging.info(f"{'='*80}\n")
    
    return accuracy_df, csv_filename


if __name__=="__main__":
    import pandas as pd
    import os
    
    # Create timestamped results folder
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_folder = f'./results_logs/results/{timestamp}'
    os.makedirs(results_folder, exist_ok=True)
    
    # Setup logging
    log_file = f'{results_folder}/eval_perf_new_{timestamp}.log'
    setup_logging(log_file)
    
    logging.info("="*80)
    logging.info("Starting Evaluation Script")
    logging.info(f"Results folder: {results_folder}")
    logging.info(f"Log file: {log_file}")
    logging.info("="*80)
    
    # Set random seed for reproducibility
    random.seed(4)
    np.random.seed(4)

    # Load human evaluation data
    merged_human_eval = pd.DataFrame()
    reviewers = ['reviewer_1', 'reviewer_2', 'reviewer_3', 'reviewer_4']
    for i, reviewer in enumerate(reviewers):
        data = pd.read_csv(f'./datasets/human_eval/{reviewer}_eval.csv')
        if i == 0:
            merged_human_eval = data.copy()
        merged_human_eval[f'{reviewer}_rating'] = data['Rating'].apply(lambda x: int(x.split()[0]))

    ratings = np.array([merged_human_eval[f'{r}_rating'].to_list() for r in reviewers])

    # Convert None to np.nan
    ratings = np.where(ratings == None, np.nan, ratings)

    # Compute Krippendorff's Alpha (nominal metric for categorical data)
    alpha = krippendorff.alpha(reliability_data=ratings, level_of_measurement="nominal")
    logging.info(f"Krippendorff's Alpha: {alpha:.4f}")

    # merge all the ratings together
    merged_human_eval['final rating']=0

    def get_final_rating(row):
        unique_values = row.value_counts()
        if unique_values.iloc[0] > 2:
            return unique_values.index[0]
        elif unique_values.iloc[0] == 2:
            return random.choice(unique_values.index)
        return -1

    rating_cols = [f'{r}_rating' for r in reviewers]
    merged_human_eval['final_rating'] = merged_human_eval[rating_cols].apply(get_final_rating, axis=1)

    # Configuration for different QA domains
    config = {
        "ImageQA": {
            "models": ["llava-1.5-7b-hf"],
            "datasets": ["textvqa", "docvqa", "pope"]
        },
        "LanguageQA": {
            "models": ["gpt4o"],
            "datasets": ["mrqa", "hotpotqa", "musique"]
        },
        "VideoQA": {
            "models": ["qwen2_5_vl_3b_instruct"],
            "datasets": ["tgif", "msvd", "msrvtt"]
        }
    }

    base_path = "./datasets/"
    syn_model = "llama-3.2-3b-instruct"
    emb_model = "ember-v1"
    data_list = []
    dataset_size = "subset_200"
    humanidx = 0
    save_domain_results = True

    # Configuration flags
    use_synans = True
    USE_ANS_FLAG = False
    
    # Dictionary to store correlations for the table
    pearson_correlations = {}
    kendall_correlations = {}
    accuracies = {}
    weight_experiments = {}
    model_dataset_mapping = {}
    
    # Mapping for display names
    dataset_display_names = {
        'tgif': 'TGIF',
        'msvd': 'MSVD',
        'msrvtt': 'MSRVTT',
        'textvqa': 'TextVQA',
        'docvqa': 'DocVQA',
        'pope': 'POPE',
        'mrqa': 'MRQA',
        'hotpotqa': 'HotpotQA',
        'musique': 'MuSiQue'
    }
    
    for model_cat, cat_data in config.items():
        domain_data={}
        for model in cat_data["models"]:
            for data in cat_data["datasets"]:
                # extract idx in the respective files
                file_path = base_path + f'subset_200/syn_ans/syn_model-{syn_model}/{data}_{model}_data.jsonl'

                # Load the ref_data
                json_data = [json.loads(line) for line in open(file_path)]
                json_data_filtered = []
                for i, temp_data in enumerate(json_data[:25]):
                    qid = temp_data['question_id' if 'question_id' in temp_data.keys() else 'id']
                    
                    if str(merged_human_eval.loc[humanidx+i, 'question_id']) in str(qid):
                        json_data_filtered.append(
                            {
                                'question_id': temp_data['question_id' if 'question_id' in temp_data.keys() else 'id'],
                                'question': temp_data['question'],
                                'answer': temp_data['answer'],
                                'pred': temp_data['pred'],
                                'syn_ans': temp_data['syn_ans'],
                                'video': temp_data.get('video', ''),
                                'human_rating': int(merged_human_eval.loc[humanidx+i, 'final_rating'])
                            }
                        )
                    else:
                        logging.error(f"QID mismatch: {data}, {qid}")
                        break
                humanidx += 25

                if USE_ANS_FLAG:
                    path_prefix = f'./evaluations/{dataset_size}/gts/{data}/{model}'
                else:
                    path_prefix = f'./evaluations/{dataset_size}/syn_model-{syn_model}/emb_model-{emb_model}/{data}/{model}'

                data_paths = {
                    'smile_score': f'./evaluations/{dataset_size}/no_syn_model/emb_model-ember-v1/{data}/{model}/{data}_smile_ans.pkl',
                    'gpt4o_score': f'{path_prefix}/{data}_gpt-4o_results.json',
                    'gpt3.5_score': f'{path_prefix}/{data}_gpt-3.5-turbo_results.json',
                    'bert_score': f'{path_prefix}/{data}_bert_score.pkl',
                    'rouge_score': f'{path_prefix}/{data}_rouge.pkl',
                    'meteor_score': f'{path_prefix}/{data}_meteor.pkl',
                    'exact_match': f'{path_prefix}/{data}_exact_match.pkl',
                    'sbert_score': f'{path_prefix}/{data}_sbert.pkl',
                }
                
                for metric in ['bleurt', 'moverscore']:
                    data_paths[f'{metric}_score'] = f'{path_prefix}/{data}_{metric}.pkl'
                
                # load the data
                dataset_params = load_eval_data(
                    ref_dataset = json_data_filtered,
                    data_paths = data_paths,
                    rerange_smile_vals = True,
                    use_human_scores = True,
                    split_qa_types = None,
                    use_synans = use_synans)
                
                logging.info(f"{model}: {data}, samples: {len(json_data_filtered)}, SMILE shape: {dataset_params['smile_score']['avg'].shape}")
                
                # Evaluate each dataset individually
                scores = eval_data(json_data_filtered, data, model, data_paths=data_paths, use_human_scores=True, smile_wt_exps=False, rerange_smile_vals=True, get_acc=True, split_qa_types=None, use_synans=use_synans, **dataset_params)
                data_list.append({'ref_data': json_data_filtered, 'scores': scores})
                
                # Extract correlations from the scores returned by eval_data
                dataset_name = dataset_display_names.get(data, data.upper())
                
                if 'pearson_correlations' in scores:
                    pearson_correlations[dataset_name] = scores['pearson_correlations']
                
                if 'kendall_correlations' in scores:
                    kendall_correlations[dataset_name] = scores['kendall_correlations']
                
                if 'accuracies' in scores:
                    accuracies[dataset_name] = scores['accuracies']
                
                # Also accumulate for domain-level analysis if needed
                domain_data = merge_domain_data(domain_data, json_data_filtered, dataset_params)
    
    # Create and export correlation tables
    pearson_df, kendall_df, csv_pearson, csv_kendall = create_and_export_correlation_tables(
        pearson_correlations, 
        kendall_correlations,
        output_path=f'{results_folder}/gpteval_synans_',
        print_tables=False
    )
    
    # Create and export accuracy table
    accuracy_df, accuracy_csv = create_and_export_accuracy_table(
        accuracies,
        output_path=f'{results_folder}/gpteval_synans_',
        print_table=False
    )
    
    logging.info("="*80)
    logging.info("Evaluation Complete!")
    logging.info("="*80)
    
    # Print final results
    logging.info("\n" + "="*120)
    logging.info("FINAL RESULTS - Pearson Correlation")
    logging.info("="*120)
    logging.info(pearson_df.to_string(float_format=lambda x: f'{x:.3f}' if pd.notna(x) else 'nan'))
    logging.info("\n")
    
    logging.info("\n" + "="*120)
    logging.info("FINAL RESULTS - Kendall's Tau")
    logging.info("="*120)
    logging.info(kendall_df.to_string(float_format=lambda x: f'{x:.3f}' if pd.notna(x) else 'nan'))
    logging.info("\n")

