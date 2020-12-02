from typing import List
from abc import abstractmethod
from typing import List, Set, Mapping
from index.structure import TermOccurrence
from index.structure import Index
import math
from enum import Enum


class IndexPreComputedVals:
    def __init__(self, index: Index):
        self.document_norm = {}
        self.index = index
        self.doc_count = None
        self.precompute_vals()

    def precompute_vals(self):
        """
            Inicializa os atributos por meio do indice (idx):
            doc_count: o numero de documentos que o indice possui
            document_norm: A norma por documento (cada termo é presentado pelo seu peso (tfxidf))
        """

        if not self.doc_count:
            self.doc_count = self.index.document_count

        for doc_id, doc in self.calc_weights().items():
            self.document_norm[doc_id] = math.sqrt(doc)

    def calc_weights(self):
        weights = {}
        for term, tfp in self.index.dic_index.items():
            print(f"term {term}")
            for to in self.index.get_occurrence_list(term):
                if to.doc_id in weights:
                    weights[to.doc_id] = \
                        VectorRankingModel.tf_idf(
                            self.doc_count, to.term_freq, tfp.doc_count_with_term) ** 2 + \
                        weights[to.doc_id]
                else:
                    weights[to.doc_id] = VectorRankingModel.tf_idf(
                        self.doc_count, to.term_freq, tfp.doc_count_with_term) ** 2

        return weights


class RankingModel:
    @abstractmethod
    def get_ordered_docs(self, query: Mapping[str, TermOccurrence],
                         docs_occur_per_term: Mapping[str, List[TermOccurrence]]) -> (List[int], Mapping[int, float]):
        raise NotImplementedError("Voce deve criar uma subclasse e a mesma deve sobrepor este método")

    def rank_document_ids(self, documents_weight):
        doc_ids = list(documents_weight.keys())
        doc_ids.sort(key=lambda x: -documents_weight[x])
        return doc_ids


class OPERATOR(Enum):
    AND = 1
    OR = 2


# Atividade 1
class BooleanRankingModel(RankingModel):
    def __init__(self, operator: OPERATOR):
        self.operator = operator

    def intersection_all(self, map_lst_occurrences: Mapping[str, List[TermOccurrence]]) -> List[int]:
        print(f"map_lst_occurrences {map_lst_occurrences}")
        set_ids = set()
        for term_1, list_terms in map_lst_occurrences.items():
            for term_2, another_list in map_lst_occurrences.items():
                if term_1 != term_2:
                    for to in list_terms:
                        if self.exist_to_doc_id(another_list, to):
                            set_ids.add(to.doc_id)
        return set_ids

    def exist_to_doc_id(self, lst_ocurrences: List[TermOccurrence], term_occurrence: TermOccurrence) -> bool:
        for to in lst_ocurrences:
            if to.doc_id == term_occurrence.doc_id:
                return True

        return False

    def union_all(self, map_lst_occurrences: Mapping[str, List[TermOccurrence]]) -> List[int]:
        set_ids = set()

        for term_list in map_lst_occurrences.values():
            for term in term_list:
                set_ids.add(term.doc_id)

        return set_ids

    def get_ordered_docs(self, query: Mapping[str, TermOccurrence],
                         map_lst_occurrences: Mapping[str, List[TermOccurrence]]) -> (List[int], Mapping[int, float]):
        """Considere que map_lst_occurrences possui as ocorrencias apenas dos termos que existem na consulta"""
        if self.operator == OPERATOR.AND:
            return self.intersection_all(map_lst_occurrences), None
        else:
            return self.union_all(map_lst_occurrences), None


# Atividade 2
class VectorRankingModel(RankingModel):

    def __init__(self, idx_pre_comp_vals: IndexPreComputedVals):
        self.idx_pre_comp_vals = idx_pre_comp_vals

    @staticmethod
    def tf(freq_term: int) -> float:
        return 1 + math.log2(freq_term)

    @staticmethod
    def idf(doc_count: int, num_docs_with_term: int) -> float:
        return math.log2(doc_count / num_docs_with_term)

    @staticmethod
    def tf_idf(doc_count: int, freq_term: int, num_docs_with_term) -> float:
        return VectorRankingModel.tf(freq_term) * VectorRankingModel.idf(doc_count,
                                                                         num_docs_with_term) if num_docs_with_term > 0 else 0

    @staticmethod
    def doc_count_with_term(list_oc):
        set_ids = set()
        for oc in list_oc:
            set_ids.add(oc.doc_id)
        return len(set_ids)

    def get_ordered_docs(self, query: Mapping[str, TermOccurrence],
                         docs_occur_per_term: Mapping[str, List[TermOccurrence]]) -> (List[int], Mapping[int, float]):

        documents_weight_original = self.calc_weights(docs_occur_per_term)

        documents_weight_query = self.calc_weights_query(query, docs_occur_per_term)

        documents_weight_original_norm = {}

        for key, norm in self.idx_pre_comp_vals.document_norm.items():
            aux_weight = 0
            for term, dw in documents_weight_original.items():
                wij = dw[key] if key in dw else 0
                wiq = documents_weight_query[term] if term in documents_weight_query else 0
                aux_weight = aux_weight + (wij * wiq)
            documents_weight_original_norm[key] = aux_weight / norm

        keys = []
        for key, dw in documents_weight_original_norm.items():
            if dw == 0:
                keys.append(key)
        for k in keys:
            del documents_weight_original_norm[k]

        documents_weight = documents_weight_original_norm

        return self.rank_document_ids(documents_weight), documents_weight

    def calc_weights(self, docs_occur_per_term):
        weights = {}
        for key, tos in docs_occur_per_term.items():
            weights[key] = {}
            for to in tos:
                weights[key][to.doc_id] = self.tf_idf(self.idx_pre_comp_vals.doc_count, to.term_freq,
                                                      self.doc_count_with_term(tos))
        return weights

    def calc_weights_query(self, query, docs_occur_per_term):
        weights_query = {}
        for key, q in query.items():
            if key in docs_occur_per_term:
                weights_query[key] = self.tf_idf(self.idx_pre_comp_vals.doc_count, q.term_freq,
                                                 self.doc_count_with_term(docs_occur_per_term[key]))
            else:
                return {}
        return weights_query
