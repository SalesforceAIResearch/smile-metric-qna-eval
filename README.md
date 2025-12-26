# SMILE: A Composite Lexical-Semantic Metric for Question-Answering Evaluation

[![arXiv](https://img.shields.io/badge/arXiv-2406.XXXX-blue.svg)](https://arxiv.org/abs/2406.XXXX)

This repository provides an implementation of **SMILE: Semantic Metric Integrating Lexical Exactness**, a novel metric for evaluating natural language generation.
## What is SMILE?
SMILE is a lightweight and reliable evaluation metric for textual and visual question answering tasks. Unlike traditional metrics like ROUGE, METEOR, and Exact Match that focus purely on lexical overlap, or embedding-based metrics like BERTScore that overlook lexical precision, SMILE strikes a balance by combining sentence-level semantics, keyword-level understanding, and exact lexical matching. This hybrid approach offers a more comprehensive and interpretable evaluation, aligning closely with human judgment while avoiding the cost, bias, and inconsistency often associated with LLM-based metrics.

## Directory Structure

```
smile-metric-qna-eval/
├── smile/
│   ├── __init__.py
│   └── smile.py                 # Core SMILE implementation
├── pyscripts/
│   ├── generate_scores.py       # Main scoring script for all metrics
│   ├── generate_syn_ans.py      # Synthetic answer generation
│   ├── eval_perf.py             # Correlation analysis and evaluation
│   ├── eval_gpt_perf.py         # GPT-based evaluation script
│   ├── utils.py                 # Utility functions
│   ├── conversations.py         # LLM conversation templates
│   └── view_results.py          # Results visualization
├── scripts/
│   ├── example_generate_scores.sh   # Example: Generate SMILE scores
│   ├── example_syn_ans.sh           # Example: Generate synthetic answers
│   └── example_gpt_eval.sh          # Example: Run GPT-based evaluation
├── datasets/
│   ├── full_set/                # Full evaluation datasets
│   │   └── syn_ans/
│   │       └── syn_model-llama-3.2-3b-instruct/
│   │           └── *.jsonl
│   ├── subset_200/              # 200-sample subsets for quick experiments
│   │   └── syn_ans/
│   │       └── syn_model-llama-3.2-3b-instruct/
│   │           └── *.jsonl
│   └── human_eval/              # Human evaluation annotations
│       └── reviewer_*.csv
├── sample_data/
│   └── sample_input.json        # Sample input for quick testing
├── smile_sample_usage.py        # Quick-start sample script
├── requirements.txt
├── pyproject.toml
└── README.md
```

## Installation

Clone this repository and install the dependencies:

```bash
git clone git@github.com:SalesforceAIResearch/smile-metric-qna-eval.git
cd smile-metric-qna-eval
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Alternatively, you can also install using pip:
```bash
pip install git+https://github.com/SalesforceAIResearch/smile-metric-qna-eval.git
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

## Supported Metrics

| Metric | Description |
|--------|-------------|
| **SMILE** | Our proposed composite lexical-semantic metric |
| **ROUGE-L** | Longest common subsequence F1 |
| **BERTScore** | Contextual embedding similarity |
| **METEOR** | Token-level matching with synonyms |
| **Exact Match** | Exact string matching |
| **sBERT** | Sentence-BERT cosine similarity |
| **BLEURT** | Learned evaluation metric |
| **MoverScore** | Earth mover's distance with BERT embeddings |
| **GPT-3.5/GPT-4o** | LLM-as-judge evaluation |

## Configuration Options

### Embedding Models
- `ember-v1`: Default embedding model for SMILE

### Synthetic Answer Models
- `llama-3.2-3b-instruct`: Local Llama model (used in provided datasets)
- `gpt-3.5-turbo`: OpenAI GPT-3.5 (requires API key)

## Datasets

The sample datasets include synthetic answers generated using Llama-3.2-3B-Instruct for:

| Category | Datasets |
|----------|----------|
| **Language QA** | HotpotQA, MRQA, MuSiQue, NaturalQuestions, TriviaQA |
| **Image QA** | DocVQA, TextVQA, POPE |
| **Video QA** | TGIF, MSVD, MSRVTT |

## Notes

1. All paths in the scripts are relative and should work from the package root directory.
2. GPU is recommended for faster embedding generation.
3. For GPT-based evaluation, you need to provide your own OpenAI API key.
4. The `subset_200` contains 200 samples per dataset for faster experimentation.

## Troubleshooting

### MoverScore Compatibility Issues

If you're using MoverScore (`--eval_mode moverscore`) and encounter errors, you may need to patch the installed `moverscore_v2.py` file:

#### Issue 1: CUDA Device Error
```
AssertionError: Torch not compiled with CUDA enabled
```
**Cause:** The library hardcodes `device = 'cuda'`, failing on machines without CUDA (e.g., macOS).

#### Issue 2: NumPy `np.float` Deprecation
```
AttributeError: module 'numpy' has no attribute 'float'
```
**Cause:** `np.float` was deprecated in NumPy 1.20 and removed in NumPy 2.0.

#### Fix (one-liner)
Run this command to patch both issues:
```bash
sed -i '' -e "s/^device = 'cuda'$/device = 'cuda' if torch.cuda.is_available() else 'cpu'/" \
          -e 's/np\.float)/float)/g' \
    .venv/lib/python3.11/site-packages/moverscore_v2.py
```
> **Note:** Adjust the path based on your Python version and virtual environment location.

## Citation

If you use this code or the SMILE metric in your research, please cite:

```
@inproceedings{smile2025,
  title={SMILE: A Composite Lexical-Semantic Metric for Question-Answering Evaluation},
  author={...},
  booktitle={Proceedings of ARR 2025},
  year={2025},
  url={https://arxiv.org/abs/2406.XXXX}
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

> **Note**: This release is for research purposes only. This release should not be used to develop models that compete with OpenAI. This release should not be used to improve any other large language model (excluding Llama 2 or derivative works thereof).

## Contributors

- [Shrikant Kendre](https://github.com/shriawesome)
- [Austin Xu]()
- [Juan Carlos Niebles]()
- [Shafiq Rayhan Joty]()
- [Honglu Zhu]()
- [Michael Ryoo]()

We welcome contributions! Please open an issue or pull request.

**For more details, see the [paper on arXiv](https://arxiv.org/abs/2406.XXXX).**
