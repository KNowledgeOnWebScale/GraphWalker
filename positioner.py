from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from agentic_base import *


class PositionerAgent(BaseClientAgent):

    def __init__(self, introduction=""):
        super().__init__('positioner')
        self.introduction = introduction

        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        self.model = SentenceTransformer(model_name, device='cpu')

        self.triple_embeddings_document = pd.read_parquet("kb_embeddings.parquet")

        self.doc_to_index = {}
        for idx in self.triple_embeddings_document.index:
            triple = self.triple_embeddings_document.loc[idx]["corpus"]
            s, p, o = triple.split(" ")[0].split("/")[-1], triple.split(" ")[1].split("/")[-1], \
            triple.split(" ")[2].split("/")[-1]
            self.doc_to_index[(s, p, o)] = idx

    def sim_search_for_docs(self, question, docs, top_k=5):
        relevant_idx = [self.doc_to_index[doc] for doc in docs]
        triple_embeddings = self.triple_embeddings_document.iloc[relevant_idx].filter(like="embedding_", axis=1)

        qemb = self.model.encode(question)

        sim_documents = cosine_similarity([qemb], triple_embeddings)
        sim_documents = sim_documents[0]
        relevant_index = np.argsort(sim_documents)[-top_k:].tolist()
        relevant_index = [self.doc_to_index[doc] for i, doc in enumerate(docs) if i in relevant_index]
        retrieved_documents = self.triple_embeddings_document.iloc[relevant_index]
        docs = retrieved_documents["corpus"].str.cat(sep=";")
        return docs

    def sim_search(self, question, top_k=5):
        triple_embeddings = self.triple_embeddings_document.filter(like="embedding_", axis=1)

        qemb = self.model.encode(question)

        sim_documents = cosine_similarity([qemb], triple_embeddings)
        sim_documents = sim_documents[0]
        relevant_index = np.argsort(sim_documents)[-top_k:].tolist()
        retrieved_documents = self.triple_embeddings_document.iloc[relevant_index]
        docs = retrieved_documents["corpus"].str.cat(sep=";")
        return docs

    def embed_kb(self, data_path: str, output_path: str) -> None:
        """Generates embeddings for triples using a vanilla RAG (Retrieval-Augmented Generation) approach.

        This function reads a dataset containing knowledge graph triples from a parquet file,
        constructs textual representations of each triple by concatenating the subject, predicate,
        and object, and then computes embeddings using the `OllamaEmbeddings` model. The resulting
        embeddings are saved in a parquet file.

        Args:
            data_path (str): Path to the input parquet file containing knowledge graph triples.
            output_path (str): Directory path where the computed embeddings should be saved.

        Outputs:
            - A parquet file containing embeddings for each triple, saved as
              `vanillaRAG_embedding_<parameters>.parquet`.

        Processing Steps:
            1. Read the input parquet file containing knowledge graph triples.
            2. Construct a textual representation for each triple by concatenating the `subject`,
               `predicate`, and `object` fields.
            3. Compute embeddings for the generated text representations using `OllamaEmbeddings`.
            4. Save the resulting embeddings as a parquet file.

        The output filename is determined based on the input file name structure, extracting
        parameters from the file name.
        """
        data_representation_path = Path(data_path)
        parameters = "_".join(data_representation_path.stem.split("_")[1:])
        kg_corpus = pd.read_parquet(data_path)
        kg_corpus["corpus"] = kg_corpus["subject"] + " " + kg_corpus["predicate"] + " " + kg_corpus["object"]
        triple_corpus_list = kg_corpus["corpus"].to_list()

        triple_vector_representation = self.model.encode(triple_corpus_list, show_progress_bar=True,
                                                         convert_to_tensor=True)

        column_names = [f"embedding_{i}" for i in range(triple_vector_representation.shape[1])]
        triple_vector_representation = pd.DataFrame(triple_vector_representation, columns=column_names)
        triple_vector_representation = pd.concat([kg_corpus, triple_vector_representation], axis=1)
        triple_vector_representation.to_parquet(f"{output_path}/kb_embeddings_{parameters}.parquet")
