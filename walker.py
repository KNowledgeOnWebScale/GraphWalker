import json
from agentic_base import *
from agentic_utils import *


class WalkerAgent(BaseClientAgent):
    def __init__(self, introduction=""):
        super().__init__('walker')
        self.introduction = introduction

    def validate(self, response, file_handler=None):
        if file_handler:
            file_handler.write("#### Walker Response ####\n")
            file_handler.write(response + "\n")

        match = CustomJSONValidator.validate(response)

        if not match:
            return False, "No JSON section found in response."

        try:
            # second pass
            CustomJSONValidator.pydantic_validate(response, WalkerResponse)
            json_obj = json.loads(match.group(1))
            options = json_obj.get("option")
            if isinstance(options, int):
                return True, options
            else:
                return False, "Expected 'option' to be a list of integers: " + str(json_obj)
        except json.JSONDecodeError:
            return False, "Invalid JSON format: " + match.group(1)

    def act(self, **kwargs):
        question = kwargs.get('question')
        options_text = kwargs.get('options_text')
        current = kwargs.get('current')

        prompt = generate_walker_prompt(introduction=self.introduction,
                                        question=question,
                                        options_text=options_text,
                                        current=current)

        file_handler = kwargs.get('file_handler')

        if file_handler:
            file_handler.write("**** Walker prompt ****\n")
            file_handler.write(prompt + "\n")

        iter_count = 0
        augmented_prompt = prompt
        while iter_count < MAX_ITER:
            # output = self.ask(augmented_prompt, WalkerResponse)
            output = self.ask(augmented_prompt)
            valid, outcome = self.validate(output, file_handler=file_handler)
            if valid:
                return outcome
            augmented_prompt = generate_failed_prompt(output, outcome, self.name)
            iter_count += 1
        # failure to walk in any direction
        return 0