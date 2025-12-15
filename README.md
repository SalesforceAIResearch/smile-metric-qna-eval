# SMILE: A Composite Lexical-Semantic Metric for Question-Answering Evaluation

This repository provides an implementation of **SMILE: Semantic Metric Integrating Lexical Exactness**, a novel metric for evaluating natural language generation.

## What is SMILE?
SMILE is a lightweight and reliable evaluation metric for textual and visual question answering tasks. Unlike traditional metrics like ROUGE, METEOR, and Exact Match that focus purely on lexical overlap, or embedding-based metrics like BERTScore that overlook lexical precision, SMILE strikes a balance by combining sentence-level semantics, keyword-level understanding, and exact lexical matching. This hybrid approach offers a more comprehensive and interpretable evaluation, aligning closely with human judgment while avoiding the cost, bias, and inconsistency often associated with LLM-based metrics.

## Installation

Clone this repository and install the dependencies:

```bash
git clone <repository-url>
cd smile-metric-qna-eval
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```



## Quick Run

Run the included sample script from the repository root to quickly verify your setup:

```bash
python3 smile_sample_usage.py
```

What this does:
- Loads the sample data from `sample_data/sample_input.json`
- Initializes SMILE with default settings (e.g., `ember-v1`, exact matching on)
- Computes scores and prints a concise summary

Example output (values will vary by environment/models):

```text
================================================================
Step 1: Loading sample input 📥
================================================================
Loaded rows: 3 📦

================================================================
Step 2: Initializing SMILE 🙂
================================================================

================================================================
Step 3: Computing SMILE scores 🧮⚙️
================================================================

================================================================
Step 4: Results summary 📊
================================================================
Sentence embedding score (mean): 0.8421 ✨
Keyword score (mean): 0.7667 🔑
SMILE avg (mean): 0.8044 😊
SMILE hm  (mean): 0.7923 🤝

-- First item details 🔎 --
question: What is the capital of France?
answer  : Paris
syn_ans : The capital of France is Paris.
pred    : Paris is known to the capital of France.
sent_emb_score: 0.8512 ✨
kwd_score     : 0.7500 🔑
avg           : 0.8006 😊
hm            : 0.7952 🤝
```

## Usage

You can use SMILE as a Python library or from the command line.

### Input Data Format
The input data for the evaluation script should in in JSON or JSONL format. Each entry in the file should be a dictionary containing the following keys:
- **id** or **question_id**: A unique identifier for the question.
- **question**: The question text.
- **answer**: The ground-truth answer(s) for the question. This can be a string list of strings (for multiple references).
- **syn_ans**: Synthetic answers generated for the question against each answer(s). Not required in case `use_ans` flag is set.
- **pred**: The predicted answer(s) for the question.
```json
{
  "id": "1",
  "question": "What is the capital of France?",
  "answer": "Paris",
  "syn_ans": "The capital of France is Paris.",
  "pred": "Paris is known to the capital of France." 
}
```

### Python API

```python
from smile.smile import SMILE
import sys

# Ensure we can import from pyscripts even when running from repo root
sys.path.append(str(Path(__file__).resolve().parent / "pyscripts"))
from generate_scores import load_data

# Example: evaluating a list of predictions against references - using the above input data format
input_path = "sample_data/sample_input.json"
args = SimpleNamespace(
        input_file=str(input_path),
        pred_file=None, # to be used if predictions are present in a separate file
        use_ans=False,
    )

proc_data = load_data(args)

# metrics to be computed - avg(average), hm(harmonic mean)
eval_metrics = ['avg', 'hm']
smile_obj = SMILE(emb_model = 'ember-v1',
                  eval_metrics = eval_metrics, 
                  assign_bins = <True/ False>, 
                  use_exact_matching = <True/ False>, 
                  save_emb_folder = <save emb folder path>, 
                  load_emb_folder = <load emb folder path>, 
                  syn_ans_model = <synthetic answer generation model name>, 
                  verbose = <True/ False>)
# When synthetic answer and ground-truth is a string
results = smile_obj.generate_scores(proc_data)
print(f"SMILE Score: {results}")
```

### Using generate_scores.py
The `generate_scores.py` script is a versatile tool for evaluating predictions against references using various metrics. It supports the following evaluation modes: SMILE, ROUGE, BERTScore, METEOR, Exact Match and sBERT.
#### Generating SMILE Scores
To compute SMILE scores, you can use the `--eval_mode`(default "smile"). The script automatically handles extracting the relevant keys from the input file and processes the data for evaluation.
```bash
python3 pyscripts/generate_scores.py \
      --input_file path/to/input.json(l) \
      --output_file path/to/output.pkl \
      --eval_mode smile \
      --timeit
```
> **Note**: You can set `--pred_file` in-case your predictions(i.e `pred`) are present in another file.


## Citation

If you use this code or the SMILE metric in your research, please cite:

```
@inproceedings{smile2025,
  title={SMILE: A Composite Lexical-Semantic Metric for Question-Answering Evaluation},
  author={Anonymous},
  year={2025}
}
```

## License

This project is licensed under the **Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)** license. 

You are free to:
- **Share**: Copy and redistribute the material in any medium or format.
- **Adapt**: Remix, transform, and build upon the material.

Under the following terms:
- **Attribution**: You must give appropriate credit, provide a link to the license, and indicate if changes were made. You may do so in any reasonable manner, but not in any way that suggests the licensor endorses you or your use.
- **NonCommercial**: You may not use the material for commercial purposes.

For more details, see the [full license text](https://creativecommons.org/licenses/by-nc/4.0/).

> **Note**: This release is for research purposes only.
