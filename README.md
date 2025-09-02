# SMILE: A Composite Lexical-Semantic Metric for Question-Answering Evaluation

[![arXiv](https://img.shields.io/badge/arXiv-2406.XXXX-blue.svg)](https://arxiv.org/abs/2406.XXXX)

This repository provides an implementation of **SMILE: Semantic Metric Integrating Lexical Exactness**, a novel metric for evaluating natural language generation.
## What is SMILE?
SMILE is a lightweight and reliable evaluation metric for textual and visual question answering tasks. Unlike traditional metrics like ROUGE, METEOR, and Exact Match that focus purely on lexical overlap, or embedding-based metrics like BERTScore that overlook lexical precision, SMILE strikes a balance by combining sentence-level semantics, keyword-level understanding, and exact lexical matching. This hybrid approach offers a more comprehensive and interpretable evaluation, aligning closely with human judgment while avoiding the cost, bias, and inconsistency often associated with LLM-based metrics.

## Installation

Clone this repository and install the dependencies:

```bash
git clone git@github.com:SalesforceAIResearch/smile-metric-qna-eval.git
cd smile-metric-qna-eval
pip install -r requirements.txt
```

Alternatively, you can also install using pip:
```bash
pip install git+https://github.com/SalesforceAIResearch/smile-metric-qna-eval.git
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

# Example: evaluating a list of predictions against references
predictions = ["The cat sat on the mat.", "A quick brown fox jumps over the lazy dog."]
references = ["A cat is sitting on the mat.", "The quick brown fox jumps over a lazy dog."]
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
