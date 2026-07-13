from typing import Optional

from rdflib import Graph, URIRef

from agentic_utils import *


class ResolverAgent:
    def __init__(self, positioning=False, filtering=False):
        self.g = Graph()
        self.g.parse("kb.ttl", format="ttl")

        self.positioning = positioning
        self.filtering = filtering

    def get_label(self, node):
        """Readable label from URIRef or Literal"""
        if isinstance(node, URIRef):
            return node.split("/")[-1]
        return str(node)

    def extract_entity(self, text):
        import re
        from rdflib import Namespace
        match = re.search(r'\[(.*?)\]', text)
        EX = Namespace("http://example.org/")
        current = EX[match.group(1).strip().replace(" ", "_").lower()]
        print("Extracted entity:", current)
        return current

    def entity_subject_hop(self, entity):
        predicates = list(set(self.g.predicates(subject=entity)))
        options = []

        txt_options = ""
        for i, predicate in enumerate(predicates, start=1):
            objects = list(self.g.objects(subject=entity, predicate=predicate))
            options.append((predicate, objects))

            # limit size of objects in case of very well-connected subjects...
            # ... the selection is arbitrary, so technically all elements should be discoverable at some point
            object_list = " | ".join(self.get_label(s) for s in objects[:MAX_LENGTH])
            txt_options += f"Option {i}: [{self.get_label(entity)}] → {self.get_label(predicate)} → {object_list}\n"

        return txt_options, options

    def entity_object_hop(self, entity):
        predicates = list(set(self.g.predicates(object=entity)))
        options = []

        txt_options = ""
        for i, predicate in enumerate(predicates, start=1):
            subjects = list(self.g.subjects(object=entity, predicate=predicate))
            options.append((predicate, subjects))

            # limit size of subjects in case of very well-connected objects...
            # ... the selection is arbitrary, so technically all elements should be discoverable at some point
            subject_list = " | ".join(self.get_label(s) for s in subjects[:MAX_LENGTH])
            txt_options += f"Option {i}: [{subject_list}] → {self.get_label(predicate)} → {self.get_label(entity)}\n"

        return txt_options, options

    def hop(self, entity):
        options = []
        txt_options = ""

        ###A) entity is subject
        predicates = list(set(self.g.predicates(subject=entity)))
        for i, predicate in enumerate(predicates, start=1):
            objects = list(self.g.objects(subject=entity, predicate=predicate))
            options.append(("subj", predicate, objects))

            # limit size of objects in case of very well-connected subjects...
            # ... the selection is arbitrary, so technically all elements should be discoverable at some point
            object_list = " | ".join(self.get_label(s) for s in objects[:MAX_LENGTH])
            if len(objects) > 1:
                txt_options += f"Option {i}: [{self.get_label(entity)}] → {self.get_label(predicate)} → [{object_list}]\n"
            else:
                txt_options += f"Option {i}: [{self.get_label(entity)}] → {self.get_label(predicate)} → {object_list}\n"

        ####B) entity is object
        predicates = list(set(self.g.predicates(object=entity)))
        for i, predicate in enumerate(predicates, start=1):
            subjects = list(self.g.subjects(object=entity, predicate=predicate))
            options.append(("obj", predicate, subjects))

            # limit size of subjects in case of very well-connected objects...
            # ... the selection is arbitrary, so technically all elements should be discoverable at some point
            subject_list = " | ".join(self.get_label(s) for s in subjects[:MAX_LENGTH])
            if len(subjects) > 1:
                txt_options += f"Option {i}: [{subject_list}] → {self.get_label(predicate)} → [{self.get_label(entity)}]\n"
            else:
                txt_options += f"Option {i}: {subject_list} → {self.get_label(predicate)} → [{self.get_label(entity)}]\n"

        return txt_options, options

    def resolve_walks(self, question, entity, walker, rephraser, positioner, file_handler, *,
                      iter_count: int = 0, max_iter: int = 4,
                      accumulated_walks: Optional[list] = None,
                      initial_search: Optional[bool] = False) -> list:
        """
        Collects all walks needed to answer *question*.
        Uses a while‑loop controlled by the rephraser:
          – option == 1  ➜ answer found ➜ stop
          – option == 2  ➜ recurse into sub‑questions, then re‑check
        """
        reply = None
        answered_sub_qs = {}
        if accumulated_walks is None:
            accumulated_walks = []

        # TODO: use LLM to check whether initial walks are relevant to the question;
        #  if not, extract entity (based on entities in initial triples) and fall back on walker
        if initial_search:
            initial_triples = positioner.sim_search(question).split(";")
            for triple in initial_triples:
                s, p, o = triple.split(" ")
                accumulated_walks.append((self.get_label(URIRef(s)),
                                          self.get_label(URIRef(p)),
                                          self.get_label(URIRef(o))))

            print("After initial search:", accumulated_walks)

        while iter_count < max_iter:
            iter_count += 1

            # ---------- 1. expand with one hop ----------

            # skip the first walk if this is not the first iteration
            # this forces subsequent walks to always be guided by further subquestions,
            # instead of starting from scratch and not knowing where to go...
            if reply is None and not initial_search:
                res = self.hop(entity)
                # trim walks that have already been explored for the current question
                # res = self.trim_walks(question, res)
                choice = walker.act(
                    question=question,
                    current=[self.get_label(entity)],
                    options_text=res[0],
                    file_handler=file_handler
                )

                # TODO: use self-consistency to avoid situations where walker should not have quit so early
                # TODO: use positioner to try and hop to a new set of triples in case walker decides to quit
                print('walker choice:', choice)
                if choice == 0:  # walker refused / nothing chosen
                    break

                print('options to select from:', res[1])
                if choice - 1 >= len(res[1]):
                    print('selected option', choice - 1, 'exceeds list of options:', len(res[1]))
                    print('reprompting walker...')
                    continue

                selected = res[1][choice - 1]

                # differentiate between position of entity in new walks, to avoid duplicates
                new_walks = [
                    (self.get_label(entity), self.get_label(selected[1]), self.get_label(r))
                    for r in selected[2] if selected[0] == "subj"
                ]

                new_walks.extend(
                    [
                        (self.get_label(r), self.get_label(selected[1]), self.get_label(entity))
                        for r in selected[2] if selected[0] == "obj"
                    ]
                )

                print('new walks:', new_walks)

                accumulated_walks.extend(new_walks)

            # remove duplicates
            print('before duplicate removal:', len(accumulated_walks))
            accumulated_walks = list(set(accumulated_walks))[:100] # limit so that context window doesn't explode
            print('after duplicate removal:', len(accumulated_walks))

            accumulated_entities = set(
                [triple[0] for triple in accumulated_walks] + [triple[-1] for triple in accumulated_walks])

            # ---------- 2. ask the rephraser ----------

            reply = rephraser.act(question=question, walks=accumulated_walks, extra_help=answered_sub_qs,
                                  file_handler=file_handler)

            if reply["option"] == 1:
                # Sufficient information gathered → finished
                break

            if reply["option"] == 2:
                # ---------- 3. recurse into every sub‑question ----------
                if isinstance(reply.get("subquestions", ""), str):
                    subquestions = [reply.get("subquestions", "")]
                else:
                    subquestions = reply.get("subquestions", [])
                # last check
                if subquestions is None:
                    subquestions = []
                for sub_q in subquestions:
                    print("Looking into subquestion:", sub_q)
                    try:
                        sub_entity = self.extract_entity(sub_q)
                    except Exception:
                        sub_q = rephraser.refine(question=question, subquestion=sub_q, entities=accumulated_entities,
                                                 file_handler=file_handler)
                        # try again!
                        try:
                            sub_entity = self.extract_entity(sub_q)
                            # check that newly extracted entity is actually in the graph...
                            if self.hop(sub_entity)[0] == "":
                                print('Something went wrong with rephrasing... restarting this iteration...')
                                continue
                        except Exception:
                            print('Something went wrong with rephrasing... restarting this iteration...')
                            continue

                    # TODO: filter accumulated walks based on some other relevance criterion...
                    #  current idea: perform sim-search on accumulated walks and newly generated subquestion, before passing to recursive call
                    sub_accumulated_walks = accumulated_walks
                    if len(accumulated_walks) > 10 and self.filtering:
                        factor = 10
                        print("Filtering relevant walks based on new subquestion with factor:", factor)
                        print("Before filtering:", len(accumulated_walks))
                        filtered_triples = positioner.sim_search_for_docs(question=sub_q,
                                                                          docs=accumulated_walks,
                                                                          top_k=factor).split(";")
                        sub_accumulated_walks = []
                        for triple in filtered_triples:
                            s, p, o = triple.split(" ")
                            sub_accumulated_walks.append((self.get_label(URIRef(s)),
                                                          self.get_label(URIRef(p)),
                                                          self.get_label(URIRef(o))))
                        print("After filtering:", len(sub_accumulated_walks))

                    answered_sub_qs[sub_q] = self.resolve_walks(
                        sub_q,
                        sub_entity,
                        walker,
                        rephraser,
                        positioner,
                        iter_count=iter_count,
                        max_iter=max_iter,
                        accumulated_walks=sub_accumulated_walks,
                        initial_search=False,
                        file_handler=file_handler
                    )[0]
                # Loop continues: rephraser will be asked again
                continue

            # Any other option or anomaly → give up politely
            print(f"Unexpected rephraser option ({reply['option']}) — stopping.")
            break

        return reply, accumulated_walks
