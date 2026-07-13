import json
from agentic_base import *
from agentic_utils import *


class RephraserAgent(BaseClientAgent):
    def __init__(self, introduction=""):
        super().__init__('rephraser')
        self.introduction = introduction

    def validate(self, response, file_handler=None):
        if file_handler:
            file_handler.write("#### Rephraser Response ####\n")
            file_handler.write(response + "\n")

        match = CustomJSONValidator.validate(response)

        if not match:
            return False, "No JSON section found in response."

        try:
            # second pass
            CustomJSONValidator.pydantic_validate(response, RephraserResponse)
            json_obj = json.loads(match.group(1))
            options = json_obj.get("option")
            if isinstance(options, int):
                if options == 2:
                    return True, {'option': 2, 'subquestions': json_obj.get('subquestions')}
                else:
                    return True, {'option': 1, 'answer': json_obj.get('answer')}
            else:
                return False, "Expected 'option' to be a list of integers: " + str(json_obj)
        except json.JSONDecodeError:
            return False, "Invalid JSON format: " + match.group(1)

    def act(self, **kwargs):
        print('Trying to resolve question based on existing walks...')

        question = kwargs.get('question')
        walks = kwargs.get('walks')
        extra_help = kwargs.get('extra_help')

        qas = ""
        for q in extra_help:
            if extra_help[q] is not None and 'answer' in extra_help[q]:
                answer = extra_help[q]["answer"]
                if isinstance(answer, list):
                    if all(isinstance(el, str) for el in answer):
                        answer = ', '.join(answer)
                    else:
                        try:
                            # try to convert...
                            answer = [str(el) for el in answer]
                            answer = ', '.join(answer)
                        except Exception:
                            answer = ''
                elif not isinstance(answer, str):
                    try:
                        # try to convert...
                        answer = str(answer)
                    except Exception:
                        answer = ''
                qas += "Question: " + q + " | Answer: " + answer + '\n'

        # add extra information about subquestions that have already been answered
        intro_text = self.introduction
        if qas != "":
            intro_text = f"""
        {self.introduction}
        Note that the following questions and answers are already known:
        {qas}
        """

        prompt = generate_rephraser_prompt(introduction=intro_text,
                                           question=question,
                                           walks=walks)

        file_handler = kwargs.get('file_handler')

        if file_handler:
            file_handler.write("#### Rephraser prompt ####\n")
            file_handler.write(prompt + "\n")

        print('Asking to see if question can be resolved...')

        iter_count = 0
        augmented_prompt = prompt
        while iter_count < MAX_ITER:
            # output = self.ask(augmented_prompt, RephraserResponse)
            output = self.ask(augmented_prompt)
            valid, outcome = self.validate(output, file_handler=file_handler)
            if valid:
                return outcome
            augmented_prompt = generate_failed_prompt(output, outcome, self.name)
            iter_count += 1
        # will lead to trying the whole loop again...
        return {"option": 2}

    def refine(self, **kwargs):
        print('Trying to refine subquestion...')

        question = kwargs.get('question')
        subquestion = kwargs.get('subquestion')
        entities = kwargs.get('entities')

        prompt = refine_subquestion_prompt(introduction=self.introduction,
                                           question=question,
                                           subquestion=subquestion,
                                           entities=entities)

        file_handler = kwargs.get('file_handler')

        if file_handler:
            file_handler.write("#### Refinement prompt ####\n")
            file_handler.write(prompt + "\n")

        print('Asking to see if subquestion can be refined...')

        # output = self.ask(prompt, RefinementResponse)
        output = self.ask(prompt)

        match = CustomJSONValidator.validate(output)

        if not match:
            return "No JSON section found in response."

        try:
            # second pass
            CustomJSONValidator.pydantic_validate(output, RefinementResponse)
            json_obj = json.loads(match.group(1))
            return json_obj.get('subquestion')
        except json.JSONDecodeError:
            return "Invalid JSON format: " + match.group(1)
