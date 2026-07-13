import re
from abc import ABC

import torch
from openai import OpenAI
from pydantic import ValidationError
from transformers import logging

from agentic_utils import *

logging.set_verbosity_debug()
print(torch.cuda.get_arch_list())

class BaseClientAgent(ABC):
    def __init__(self, name, base_url="", model=""):
        self.name = name
        self.client = OpenAI(
            base_url=base_url,
            api_key="EMPTY"
        )
        self.model = model

    def ask(self, prompt):
        print('got original prompt:', prompt, '\n\n')
        completion = self.client.chat.completions.create(
            extra_body={},
            model=self.model,
            temperature=0,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )
        print('answer generated:', completion.choices[0].message.content)
        return completion.choices[0].message.content


class CustomJSONValidator:
    @staticmethod
    def validate(response):
        # Match the "### FINAL JSON ###" section with optional code block
        match = re.search(r'''
            ### FINAL JSON ###
            :?                        # Optional colon
            \s*                       # Optional whitespace
            (?:```json\s*)?          # Optional code block start
            (\{.*?\})                # Capture JSON object
            (?:\s*```)?              # Optional code block end
        ''', response, re.DOTALL | re.VERBOSE)

        return match

    @staticmethod
    def pydantic_validate(response, typing):
        if isinstance(response, str) and isinstance(typing, BaseModel):
            try:
                return typing.model_validate_json(response)
            except ValidationError:
                return None
