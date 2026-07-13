import sys
import random

import pandas as pd
from openai import OpenAI

from agentic_utils import *
from evaluations import *
from rephraser import RephraserAgent
from resolver import ResolverAgent
from walker import WalkerAgent
from positioner import PositionerAgent

random.seed(0)

if __name__ == '__main__':
    results_folder = sys.argv[1]
    results_setting = sys.argv[2]
    question_set = sys.argv[3]

    filtering = bool(int(sys.argv[4]))
    positioning = bool(int(sys.argv[5]))

    root = os.path.join('pickles', results_folder, results_setting)

    if not os.path.exists(os.path.join(root, question_set + '_questions.pkl')) or \
            not os.path.exists(os.path.join(root, question_set + '_ground_truth.pkl')) or \
            not os.path.exists(os.path.join(root, question_set + '_predictions.pkl')):
        questions, gts, predictions = {}, {}, {}
    else:
        questions = load_from_pickle(os.path.join(root, question_set + '_questions.pkl'))
        gts = load_from_pickle(os.path.join(root, question_set + '_ground_truth.pkl'))
        predictions = load_from_pickle(os.path.join(root, question_set + '_predictions.pkl'))

    print('Loading walker...')
    walker = WalkerAgent()
    print('Walker loaded.')

    print('Loading rephraser...')
    rephraser = RephraserAgent()
    print('Rephraser loaded.')

    print('Loading positioner...')
    positioner = PositionerAgent()
    print('Positioner loaded.')

    print('Loading resolver...')
    resolver = ResolverAgent(filtering=filtering, positioning=positioning)
    print('Resolver loaded.')

    # set api endpoint
    base_url = sys.argv[6]
    model = sys.argv[7]

    walker.client = OpenAI(base_url=base_url, api_key="EMPTY")
    walker.model = model

    rephraser.client = OpenAI(base_url=base_url, api_key="EMPTY")
    rephraser.model = model

    cap = int(sys.argv[8])

    print("Resolver settings:", resolver.filtering, resolver.positioning)

    df = pd.read_csv('MetaQA/' + question_set + '/vanilla/qa_test.txt', delimiter='\t', names=['question', 'answer'])
    with open("results.txt", 'w') as results_file:

        # shuffle ids
        idx = list(random.sample(range(len(df)), len(df)))
        if not os.path.exists(os.path.join(root, question_set + '_idx.pkl')):
            dump_to_pickle(idx, os.path.join(root, question_set + '_idx.pkl'))
        else:
            idx = load_from_pickle(os.path.join(root, question_set + '_idx.pkl'))
        print('Already done:', questions.keys())

        print('Iterating over questions...')
        for i, row in tqdm(df.loc[idx].iterrows(), total=len(df.loc[idx])):
            if i not in idx or i in questions.keys():
                continue

            if -1 < cap <= len(questions.keys()):
                break

            print("Answering question with ID", i)

            question = row['question']
            gt = row['answer']

            questions[i] = question
            gts[i] = gt

            entity = resolver.extract_entity(question)
            print('Entity extracted <', entity, '> from question <', question, '>')

            with open('logs/' + '_'.join(question.split(' ')).replace('/', '') + ".txt", 'w') as f:
                print('Starting to resolve question <', question, '> with extracted entity <', entity, '>')
                x, final_walks = resolver.resolve_walks(question, entity, walker, rephraser, positioner, f, initial_search=resolver.positioning)
                print('***final answer***\n', x)
                if x is not None and 'answer' in x:
                    results_file.write(f"{question}\t{x['answer']}\n")
                    results_file.flush()

                    predictions[i] = x['answer']
                else:
                    failed_answer = "could not answer this question as no relevant information about the question was available in the knowledge base"
                    print('***failed answer***\ntrying anyway...\n')

                    if len(final_walks) > 0:
                        final_promt = generate_final_answer_prompt(question=question, context=final_walks)
                        failed_answer = rephraser.ask(final_promt)

                    results_file.write(f"{failed_answer}\n")
                    results_file.flush()

                    predictions[i] = failed_answer

            dump_to_pickle(questions, os.path.join(root, question_set + '_questions.pkl'))
            dump_to_pickle(gts, os.path.join(root, question_set + '_ground_truth.pkl'))
            dump_to_pickle(predictions, os.path.join(root, question_set + '_predictions.pkl'))

    client = OpenAI(base_url=base_url, api_key="EMPTY")
    evaluate_predictions(questions, gts, predictions, model, client)
