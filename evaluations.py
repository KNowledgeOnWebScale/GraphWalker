import json
import os
from datetime import datetime

from loguru import logger
from openai import APIConnectionError, RateLimitError
from tqdm import tqdm
from transformers import LlamaTokenizerFast

CACHE_DIR = os.path.join("hf_cache")

tokenizer = LlamaTokenizerFast.from_pretrained("meta-llama/Llama-3.1-70B", cache_dir=CACHE_DIR)

INSTRUCTIONS = """
    # Task: 
    You are given a Question, a model Prediction, and a list of Ground Truth answers, judge whether the model Prediction matches any answer from the list of Ground Truth answers. Follow the instructions step by step to make a judgement. 
    1. If the model prediction matches any provided answers from the Ground Truth Answer list, "Accuracy" should be "True"; otherwise, "Accuracy" should be "False".
    2. If the model prediction says that it couldn't answer the question or it doesn't have enough information, "Accuracy" should always be "False".
    3. If the Ground Truth is "invalid question", "Accuracy" is "True" only if the model prediction is exactly "invalid question".
    4. Determine the Hits@1 accuracy of the model by comparing the model prediction with the first element relevant to answer the question.
    # Output: 
    Respond with only JSON string with an "Accuracy" field which is "True" or "False" and a second key with a "Hits1" field.
"""

IN_CONTEXT_EXAMPLES = """
    # Examples:
    Question: how many seconds is 3 minutes 15 seconds?
    Ground truth: ["195 seconds"]
    Prediction: 3 minutes 15 seconds is 195 seconds.
    Accuracy: True

    Question: Who authored The Taming of the Shrew (published in 2002)?
    Ground truth: ["William Shakespeare", "Roma Gill"]
    Prediction: The author to The Taming of the Shrew is Roma Shakespeare.
    Accuracy: False

    Question: Who played Sheldon in Big Bang Theory?
    Ground truth: ["Jim Parsons", "Iain Armitage"]
    Prediction: I am sorry I don't know.
    Accuracy: False
"""


def attempt_api_call(client, model_name, messages, max_retries=10):
    """Attempt an API call with retries upon encountering specific errors."""
    # todo: add default response when all efforts fail
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                response_format={"type": "json_object"},
            )
            return response.choices[0].message.content
        except (APIConnectionError, RateLimitError):
            logger.warning(f"API call failed on attempt {attempt + 1}, retrying...")
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            break
    return None


def parse_response(resp: str):
    """Pass auto-eval output from the evaluator."""
    try:
        resp = resp.lower()
        model_resp = json.loads(resp)
        answer = -1
        if "accuracy" in model_resp and (
                (model_resp["accuracy"] is True)
                or (
                        isinstance(model_resp["accuracy"], str)
                        and model_resp["accuracy"].lower() == "true"
                )
        ):
            answer = 1
        else:
            raise ValueError(f"Could not parse answer from response: {model_resp}")

        return answer
    except:
        return -1


def log_response(messages, response, output_directory="api_responses"):
    """Save the response from the API to a file."""
    os.makedirs(output_directory, exist_ok=True)
    file_name = datetime.now().strftime("%d-%m-%Y-%H-%M-%S.json")
    file_path = os.path.join(output_directory, file_name)
    with open(file_path, "w") as f:
        json.dump({"messages": messages, "response": response}, f)


def trim_predictions_to_max_token_length(prediction):
    """Trims prediction output to 75 tokens using Llama2 tokenizer"""
    max_token_length = 256
    tokenized_prediction = tokenizer.encode(prediction)
    trimmed_tokenized_prediction = tokenized_prediction[1: max_token_length + 1]
    trimmed_prediction = tokenizer.decode(trimmed_tokenized_prediction)
    return trimmed_prediction


def evaluate_predictions(queries, ground_truths, predictions, evaluation_model_name, client):
    n_miss, n_correct, n_correct_exact = 0, 0, 0
    system_message = INSTRUCTIONS + IN_CONTEXT_EXAMPLES

    for _idx, pkey in enumerate(tqdm(
            predictions.keys(), total=len(predictions.keys()), desc="Evaluating Predictions"
    )):
        query = queries[pkey]
        ground_truth = ground_truths[pkey].strip()

        prediction = predictions[pkey]
        print('Evaluating prediction:', prediction)

        # parse prediction
        if not prediction:
            prediction = ''
        elif isinstance(prediction, list):
            if all(isinstance(el, str) for el in prediction):
                prediction = ', '.join(prediction)
            else:
                try:
                    # try to convert...
                    prediction = [str(el) for el in prediction]
                    prediction = ', '.join(prediction)
                except Exception:
                    prediction = ''
        elif isinstance(prediction, int):
            try:
                prediction = str(prediction)
            except Exception:
                prediction = ''

        # trim prediction to 75 tokens using Llama2 tokenizer
        prediction = trim_predictions_to_max_token_length(prediction)
        prediction = prediction.strip()

        ground_truth_lowercase = ground_truth.lower()
        prediction_lowercase = prediction.lower()

        if "could not answer this question as no relevant information about the question was available in the knowledge base" in prediction_lowercase:
            n_miss += 1
            continue
        elif prediction_lowercase == ground_truth_lowercase:
            n_correct_exact += 1
            n_correct += 1
            continue

        messages = [
            {"role": "system", "content": system_message},
            {
                "role": "user",
                "content": f"Question: {query}\n Ground truth: {ground_truth}\n Prediction: {prediction}\n",
            },
        ]

        # if previous evaluation attempts fail...
        response = attempt_api_call(client, evaluation_model_name, messages)
        if response:
            log_response(messages, response)
            eval_res = parse_response(response)
            if eval_res == 1:
                n_correct += 1

    n = len(predictions.keys())
    results = {
        "score": (2 * n_correct + n_miss) / n - 1,
        "exact_accuracy": n_correct_exact / n,
        "accuracy": n_correct / n,
        "hallucination": (n - n_correct - n_miss) / n,
        "missing": n_miss / n,
        "n_miss": n_miss,
        "n_correct": n_correct,
        "n_correct_exact": n_correct_exact,
        "total": n,
    }
    logger.info(results)
    return results
