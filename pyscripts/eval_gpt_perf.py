"""
GPT-based Evaluation Script
Evaluates predictions using GPT-3.5-turbo or GPT-4o for question-answer scoring.
"""

import openai
import os
import time
import json
import ast
import argparse
from tqdm import tqdm
from conversations import ConversationMessage
from multiprocessing.pool import Pool
import logging
from pathlib import Path

# Setup logging - only show warnings and errors by default
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress HTTP request logging from httpx and openai
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

def parse_args():
    parser = argparse.ArgumentParser(description="question-answer-generation-using-gpt-3")
    parser.add_argument("--pred_path", required=True, help="The path to file containing prediction.")
    parser.add_argument("--output_dir", required=True, help="The path to save annotation json files.")
    parser.add_argument("--output_json", required=True, help="The path to save annotation final combined json file.")
    parser.add_argument("--api_key", default="", help="OpenAI API key.")
    parser.add_argument("--api_base", default="", type=str, help="OpenAI API base.")
    parser.add_argument("--num_tasks", default=1, type=int, help="Number of splits.")
    parser.add_argument("--eval_mode", default="llm_eval", choices=["llm_eval", "bert_score"], type=str, help="Choose the matrix to run the evaluation.")
    parser.add_argument("--llm_mode", default="lqa", choices=["lqa", 'vqa'], type=str, help="LLM evaluation mode: language QA or visual QA.")
    parser.add_argument("--openai_model", default="gpt-3.5-turbo", type=str, help="OpenAI model to use")
    parser.add_argument("--timeit", action="store_true", help="Enable timing of the code execution")
    parser.add_argument("--max_retries", default=3, type=int, help="Maximum number of retries for API calls")
    parser.add_argument("--retry_delay", default=1, type=int, help="Delay between retries in seconds")
    parser.add_argument("--dataset", default="", type=str, help="Dataset name (e.g., hotpotqa, mrqa, docvqa)")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    return args


def parse_response(response_text):
    """
    Fast, robust parsing of LLM response with fallback strategies.
    Optimized for speed - tries most common formats first.
    """
    # Fast path: Try ast.literal_eval first (most common)
    try:
        return ast.literal_eval(response_text)
    except:
        pass
    
    # Fallback: Try json.loads
    try:
        return json.loads(response_text)
    except:
        pass
    
    # Last resort: Quick quote fix for malformed responses
    try:
        return json.loads(response_text.replace("'", '"'))
    except:
        pass
    
    return None


def llm_based_eval(prediction_set, caption_files, output_dir, args):
    """
    Evaluate predictions using LLM-based scoring.
    Enhanced with better error handling and retry logic.
    """
    # Set the OpenAI API key
    openai.api_key = args.api_key
    conv_message = ConversationMessage()
    
    if args.api_base:
        openai.api_base = args.api_base
    
    successful = 0
    failed = 0
    
    for file in caption_files:
        try:
            key = file[:-5]  # Strip file extension
            qa_set = prediction_set[key]
            question = qa_set['q']
            pred = qa_set['pred']
            answer = qa_set['syn_ans']
            
            # Prepare data for API call
            data = {'question': question, 'answer': answer, 'pred': pred}
            
            messages = [
                conv_message.message_template[args.llm_mode]['system'],
                {
                    "role": conv_message.message_template[args.llm_mode]['user']['role'],
                    "content": conv_message.message_template[args.llm_mode]['user']['content'].format(**data)
                }
            ]
            
            # Retry logic for API calls and parsing
            response_dict = None
            last_response = None
            
            for attempt in range(args.max_retries):
                try:
                    completion = openai.chat.completions.create(
                        model=args.openai_model,
                        messages=messages,
                        temperature=0.0,
                    )
                    
                    response_message = completion.choices[0].message.content
                    last_response = response_message  # Save for error logging
                    
                    # Use robust parser with multiple strategies
                    response_dict = parse_response(response_message)
                    
                    if response_dict:
                        break  # Success, exit retry loop
                    else:
                        # Parsing failed, retry with new API call
                        if attempt < args.max_retries - 1:
                            time.sleep(0.5)  # Shorter delay for parse errors
                            continue
                        else:
                            raise ValueError(f"Failed to parse response: {response_message[:100]}")
                    
                except openai.RateLimitError:
                    if attempt < args.max_retries - 1:
                        time.sleep(args.retry_delay * (attempt + 1))
                    else:
                        raise
                        
                except openai.APIError as e:
                    if attempt < args.max_retries - 1:
                        time.sleep(args.retry_delay)
                    else:
                        raise
                
                except Exception as e:
                    # Handle any other parsing/API errors
                    if attempt < args.max_retries - 1:
                        time.sleep(0.5)  # Quick retry for other errors
                        continue
                    else:
                        raise
            
            if response_dict:
                result_qa_pair = [response_dict, qa_set]
                
                # Save the question-answer pairs to a json file
                output_path = Path(output_dir) / f"{key}.json"
                with open(output_path, "w") as f:
                    json.dump(result_qa_pair, f, indent=2)
                
                successful += 1
            
        except Exception as e:
            failed += 1
            
            # Create error response_dict to save
            error_response_dict = {
                "score": 0,
                "pred": "no",
                "error": str(e)
            }
            result_qa_pair = [error_response_dict, qa_set]
            
            # Save the error response as a regular json file
            output_path = Path(output_dir) / f"{key}.json"
            with open(output_path, "w") as f:
                json.dump(result_qa_pair, f, indent=2)
    
    if failed > 0:
        print(f"⚠ Warning: {failed} entries failed and saved with score=0")


def annotate(prediction_set, caption_files, output_dir, args):
    """
    Evaluates question and answer pairs using GPT-3.
    Returns a score for correctness.
    """
    if args.eval_mode == 'llm_eval':
        llm_based_eval(prediction_set, caption_files, output_dir, args)


def format_time(start_time, end_time, title='Execution time'):
    """
    Format's the time in hh:mm:ss format
    """
    time_elapsed = end_time - start_time
    hours = int(time_elapsed // 3600)
    mins = int((time_elapsed % 3600) // 60)
    secs = int(time_elapsed % 60)
    print(f"{title}: {hours:02d}:{mins:02d}:{secs:02d}")


def load_predictions(pred_path):
    """
    Load predictions from file (supports .json and .jsonl formats).
    """
    if not os.path.exists(pred_path):
        raise FileNotFoundError(f"Prediction file not found: {pred_path}")
    
    if pred_path.endswith('.jsonl'):
        with open(pred_path, 'r') as f:
            contents = [json.loads(line.strip()) for line in f if line.strip()]
    elif pred_path.endswith('.json'):
        with open(pred_path, 'r') as f:
            contents = json.load(f)
    else:
        raise ValueError(f"Unsupported file format: {pred_path}. Use .json or .jsonl")
    
    print(f"Loaded {len(contents)} predictions from {os.path.basename(pred_path)}")
    return contents


def prepare_prediction_set(pred_contents):
    """
    Prepare prediction set dictionary from loaded contents.
    """
    # Detect question_id field name
    question_id_field = 'id' if pred_contents and ('id' in pred_contents[0].keys()) else 'question_id'
    
    prediction_set = {}
    for sample in pred_contents:
        q_id = str(sample[question_id_field])
        answer = sample.get('answer', '')
        question = sample.get('question', '')
        pred = sample.get('pred', '')
        syn_ans = sample.get('syn_ans', '')  # Optional synthetic answer
        
        qa_set = {
            "question_id": q_id,
            "q": question,
            "a": answer,
            "pred": pred,
            "syn_ans": syn_ans
        }
        prediction_set[q_id] = qa_set
    
    return prediction_set, question_id_field


def calculate_metrics(combined_contents, dataset_name=""):
    """
    Calculate evaluation metrics from combined results.
    Enhanced to support different dataset types.
    Ignores entries with errors (pred='invalid' or 'error' key present).
    """
    score_sum = 0
    count = 0
    yes_count = 0
    no_count = 0
    score_distribution = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    errors = []
    skipped_errors = 0
    
    for key, result in combined_contents.items():
        try:
            # Check if this is an error entry and skip it
            if 'error' in result[0]:
                skipped_errors += 1
                continue
            
            count += 1
            score_match = result[0].get('score', 0)
            score = float(score_match)
            score_sum += score
            
            # Track score distribution
            score_int = int(round(score))
            if score_int in score_distribution:
                score_distribution[score_int] += 1
            
            # Computing accuracy for yes/no questions
            pred = result[0].get('pred', '').lower()
            if "yes" in pred:
                yes_count += 1
            elif "no" in pred:
                no_count += 1
                
        except Exception as e:
            errors.append((key, str(e)))
    
    # Calculate final metrics
    metrics = {
        'total_samples': count,
        'average_score': score_sum / count if count > 0 else 0,
        'yes_count': yes_count,
        'no_count': no_count,
        'score_distribution': score_distribution,
        'errors': len(errors),
        'skipped_errors': skipped_errors
    }
    
    # Calculate accuracy if applicable
    if yes_count + no_count > 0:
        metrics['accuracy'] = yes_count / (yes_count + no_count)
    
    # Print concise metrics
    print(f"\n{'='*60}")
    print(f"RESULTS{f' - {dataset_name.upper()}' if dataset_name else ''}")
    print(f"{'='*60}")
    print(f"Evaluated: {count} samples | Skipped: {skipped_errors} errors")
    print(f"Average Score: {metrics['average_score']:.4f}")
    
    if 'accuracy' in metrics:
        print(f"Accuracy: {metrics['accuracy']:.4f} (Yes: {yes_count}, No: {no_count})")
    
    print(f"{'='*60}\n")
    
    return metrics


def main():
    """
    Main function to control the flow of the program.
    Enhanced version with better error handling and logging.
    """
    # Parse arguments
    args = parse_args()
    
    # Enable verbose logging if requested
    if args.verbose:
        logging.getLogger(__name__).setLevel(logging.INFO)
    
    print(f"\n{'='*60}")
    print(f"Evaluation Pipeline - {args.openai_model}")
    print(f"{'='*60}")
    
    overall_start = time.time()
    
    # Load predictions
    try:
        pred_contents = load_predictions(args.pred_path)
    except Exception as e:
        print(f"✗ Error: Failed to load predictions: {e}")
        return
    
    # Prepare prediction set
    prediction_set, question_id_field = prepare_prediction_set(pred_contents)
    id_list = list(prediction_set.keys())
    caption_files = [f"{id}.json" for id in id_list]
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Start evaluation timing
    if args.timeit:
        eval_start = time.time()
    
    num_tasks = args.num_tasks
    
    # Main evaluation loop
    print(f"Processing {len(caption_files)} samples with {num_tasks} parallel tasks...\n")
    
    while True:
        try:
            # Check completed files
            completed_files = [f for f in os.listdir(output_dir) if f.endswith('.json') and not f.endswith('_error.json')]
            
            # Files that have not been processed yet
            incomplete_files = [f for f in caption_files if f not in completed_files]
            
            # Break the loop when there are no incomplete files
            if len(incomplete_files) == 0:
                print(f"✓ Completed: {len(completed_files)}/{len(caption_files)} samples")
                break
            
            # Adjust number of tasks if needed
            if len(incomplete_files) <= num_tasks:
                num_tasks = 1
            
            # Split tasks into parts
            part_len = max(1, len(incomplete_files) // num_tasks)
            all_parts = [incomplete_files[i:i + part_len] for i in range(0, len(incomplete_files), part_len)]
            task_args = [(prediction_set, part, str(output_dir), args) for part in all_parts]
            
            # Use a pool of workers to process the files in parallel
            with Pool(processes=num_tasks) as pool:
                pool.starmap(annotate, task_args)
            
        except KeyboardInterrupt:
            print("\n⚠ Interrupted by user. Exiting...")
            return
        except Exception as e:
            print(f"\n✗ Error during evaluation: {e}")
            break
    
    # Timing for evaluation
    if args.timeit:
        eval_end = time.time()
        format_time(eval_start, eval_end, title="Evaluation time")
    
    # Combine results
    if args.eval_mode == 'llm_eval':
        combined_contents = {}
        
        # Iterate through json files
        for file_name in os.listdir(output_dir):
            if file_name.endswith(".json") and not file_name.endswith("_error.json"):
                file_path = output_dir / file_name
                try:
                    with open(file_path, "r") as json_file:
                        content = json.load(json_file)
                        combined_contents[file_name[:-5]] = content
                except Exception as e:
                    print(f"✗ Error reading {file_name}: {e}")
        
        # Write combined content to output JSON
        json_path = Path(args.output_json)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(json_path, "w") as json_file:
            json.dump(combined_contents, json_file, indent=2)
        
        print(f"\n✓ Results saved: {json_path}")
        
        # Calculate and display metrics
        metrics = calculate_metrics(combined_contents, args.dataset)
        
        # Save metrics to separate file
        metrics_path = json_path.parent / f"{json_path.stem}_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
    
    # Total execution time
    if args.timeit:
        overall_end = time.time()
        format_time(overall_start, overall_end, title="Total time")
    
    print("✓ Evaluation completed\n")


if __name__ == "__main__":
    main()

