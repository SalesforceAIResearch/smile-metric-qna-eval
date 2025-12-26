"""
Conversation Message Templates for LLM-based Evaluation
Defines prompts for evaluating QA pairs using GPT models.
"""

class ConversationMessage:
    def __init__(self):
        self.message_template = {
            # General for all VQA tasks
            'vqa': {
                'system': {
                    "role": "system",
                    "content":
                        "You are an intelligent chatbot designed for evaluating the correctness of generative outputs for question-answer pairs. "
                        "Your task is to compare the predicted answer with the correct answer and determine if they match meaningfully. Here's how you can accomplish the task:"
                        "------"
                        "##INSTRUCTIONS: "
                        "- Focus on the meaningful match between the predicted answer and the correct answer.\n"
                        "- Consider synonyms or paraphrases as valid matches.\n"
                        "- Evaluate the correctness of the prediction compared to the answer."
                },
                'user': {
                    "role": "user",
                    "content":
                        "Please evaluate the following video-based question-answer pair:\n\n"
                        "Question: {question}\n"
                        "Correct Answer: {answer}\n"
                        "Predicted Answer: {pred}\n\n"
                        "Provide your evaluation only as a yes/no and score where the score is an integer value between 0 and 5, with 5 indicating the highest meaningful match. "
                        "Please generate the response in the form of a Python dictionary string with keys 'pred' and 'score', where value of 'pred' is  a string of 'yes' or 'no' and value of 'score' is in INTEGER, not STRING."
                        "DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. Only provide the Python dictionary string. "
                        "For example, your response should look like this: {{'pred': 'yes', 'score': 4.8}}."
                }
            },
            
            # Language QA evaluation template
            'lqa': {
               'system' : {
                    "role": "system",
                    "content": 
                        "You are an expert evaluator for video-based question answering systems. Your task is to judge the factual accuracy of a predicted answer by comparing it to a correct answer."
                        " You will follow a structured evaluation approach to ensure consistency:"
                        "\n\n"
                        "------\n"
                        "## INSTRUCTIONS:\n"
                        "Step 1: Extract the key facts from the Correct Answer.\n"
                        "Step 2: In case the Correct Answer is a list, choose the best answer that matches the Predicted Answer."
                        "Step 3: Extract the key facts from the Predicted Answer.\n"
                        "Step 4: Compare the two sets of facts and determine how consistent they are.\n"
                        " - Consider paraphrasing, synonyms, and partial overlaps.\n"
                        " - Ignore grammatical errors.\n"
                        " - Penalize hallucinated or contradicted information.\n"
                        "Step 4: Based on the comparison, assign a factual accuracy score between 0 and 5 (INTEGER only), where:\n"
                        " 5 = Fully accurate and aligned\n"
                        " 4 = Mostly accurate, minor omissions or paraphrasing\n"
                        " 3 = Partially correct but with notable missing or incorrect info\n"
                        " 2 = Limited accuracy, mostly incorrect or unrelated\n"
                        " 1 = Completely inaccurate\n"
                        " 0 = No relation or total hallucination\n\n"
                        "Respond strictly in the following format:\n"
                        "{'score': X, 'pred':Y} where X is an integer between 0 and 5 and Y is a either 'yes'(X>3) or 'no'(X<=3). Do not include any explanation or extra text."
                    },
                
                'user':    {
                    "role": "user",
                    "content":
                        "Evaluate the following video-based QA pair:\n\n"
                        "Question: {question}\n"
                        "Correct Answer: {answer}\n"
                        "Predicted Answer: {pred}\n\n"
                        "Return your evaluation following the instructions above."
                        "DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. Only provide the Python dictionary string. "
                    }
                }
            
        }

