import os
import pickle
from transformers import TextGenerationPipeline
from pydantic import BaseModel
from typing import List

MAX_LENGTH = 25
MAX_ITER = 4


def construct_prompt(message):
    messages = []
    messages.append({
        "role": "user",
        "content": message
    })
    return messages


def generate_failed_prompt(response, failure, agent):
    if agent == 'walker':
        expected_format = '### FINAL JSON ###\n{"option": number as int}'
    elif agent == 'rephraser':
        expected_format = '### FINAL JSON ###\n{"option": number as int, "answer": [answer as str], "subquestion": [new subquestion as str]}'
    else:
        expected_format = '### FINAL JSON ###\n{"subquestion": subquestion as str with entity in brackets}'
    return f"The following response is not correctly formatted: **{response}**.\n" \
           f"The problem with the format is: **{failure}**.\n" \
           f"Provide the same response in the following correct format:\n{expected_format}"


class RephraserResponse(BaseModel):
    option: int
    answer: List[str]
    subquestion: List[str]


def generate_rephraser_prompt(**kwargs):
    introduction = kwargs.get('introduction')
    question = kwargs.get('question')
    walks = kwargs.get('walks')

    current_walk_text = ""
    for w in walks:
        current_walk_text += " → ".join(w) + '\n'

    prompt = f"""
    The following walks (format: subject → relationship → object) represent your current knowledge base:
    {current_walk_text}

    The original question you are trying to resolve is:
    **{question}**

    You have ONLY two options:
    - 1: You can answer the original question, based on the current knowledge base.
    - 2: You can create new subquestions in natural language, mentioning ONLY one entity from the knowledge base, to make the original question more specific.

    First, SELECT either 1 or 2.

    Then, think step by step about which walk in the knowledge base brings you closer to the answer.
    {introduction}
    
    Finally, provide your selected option NUMBER (1 or 2), together with the answer (if option 1) OR new subquestions (if option 2), in the following JSON format:
    ### FINAL JSON ###
    {{"option": NUMBER, "answer": [answer], "subquestions": [list of new subquestions]}}
    """.strip()

    return prompt


class RefinementResponse(BaseModel):
    subquestion: str


def refine_subquestion_prompt(**kwargs):
    introduction = kwargs.get('introduction')
    original_question = kwargs.get('question')
    subquestion = kwargs.get('subquestion')
    entities = kwargs.get('entities')

    entities_text = ""
    for entity in entities:
        entities_text += entity + '\n'

    prompt = f"""
    {introduction}
    The original question you are trying to resolve is:
    **{original_question}**

    The new subquestion you are trying to formulate is:
    **{subquestion}**

    Make sure the subquestion contains exactly ONE of the following entities:
    {entities_text}

    Return the subquestion with the [entity] in brackets, in the following JSON format:
    ### FINAL JSON ###
    {{"subquestion": subquestion with entity in brackets}}
    """.strip()

    return prompt


class WalkerResponse(BaseModel):
    option: int


def generate_walker_prompt(**kwargs):
    introduction = kwargs.get('introduction')
    question = kwargs.get('question')
    options_text = kwargs.get('options_text')
    current = kwargs.get('current')

    current_walk_text = " → ".join(current)

    prompt = f"""
    {introduction}
    Your current position in the graph is:
    {current_walk_text}

    You are given the following options to walk along (format: subject → relationship → object):
    {options_text}

    Think step by step about which of the previous options (starting from Option 1) you should SELECT to get closer to answering the following question:
    **{question}**

    ONLY if none of the options seem in any way relevant to the question, SELECT Option 0.

    Finally, provide your SELECTED option NUMBER (e.g. 0, 1, 2) in the following JSON format:
    ### FINAL JSON ###
    {{"option": NUMBER}}
    """.strip()

    return prompt


def generate_final_answer_prompt(**kwargs):
    question = kwargs.get('question')
    context = kwargs.get('context')

    current_walk_text = ""
    for w in context:
        current_walk_text += " → ".join(w) + '\n'

    prompt = f"""
    Please provide an answer to the question: **{question}?**
    Base your answer on the following information:
    {current_walk_text}
    If you are not sure about the answer based on the available information, return: 'could not answer this question as no relevant information about the question was available in the knowledge base'
    """.strip()

    return prompt


def dump_to_pickle(data, file_path):
    """Write a binary dump to a pickle file of arbitrary size."""
    """misc: https://stackoverflow.com/questions/31468117/python-3-can-pickle-handle-byte-objects-larger-than-4gb"""
    bytes_out = pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
    with open(file_path, 'wb') as f_out:
        # 2**31 - 1 is the max nr. of bytes pickle can dump at a time
        for idx in range(0, len(bytes_out), 2 ** 31 - 1):
            f_out.write(bytes_out[idx:idx + 2 ** 31 - 1])
    return


def load_from_pickle(file_path):
    """Read from a pickle file of arbitrary size."""
    """misc: https://stackoverflow.com/questions/31468117/python-3-can-pickle-handle-byte-objects-larger-than-4gb"""
    bytes_in = bytearray(0)
    input_size = os.path.getsize(file_path)
    with open(file_path, 'rb') as f_in:
        for _ in range(0, input_size, 2 ** 31 - 1):
            bytes_in += f_in.read(2 ** 31 - 1)
    return pickle.loads(bytes_in)
