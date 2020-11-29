from typing import List, Set, Mapping
from nltk.tokenize import word_tokenize
from util.time import CheckTime
from query.ranking_models import RankingModel, VectorRankingModel, \
    BooleanRankingModel, IndexPreComputedVals, OPERATOR
from index.structure import Index, TermOccurrence
from index.indexer import Cleaner


class QueryRunner:
    def __init__(self, ranking_model: RankingModel, index: Index, cleaner: Cleaner):
        self.ranking_model = ranking_model
        self.index = index
        self.cleaner = cleaner

    def get_relevance_per_query(self) -> Mapping[str, Set[int]]:
        """
		Adiciona a lista de documentos relevantes para um determinada query (os documentos relevantes foram
		fornecidos no ".dat" correspondente. Por ex, belo_horizonte.dat possui os documentos relevantes da consulta "Belo Horizonte"

		"""
        dic_relevance_docs = {}

        for arquiv in ["belo_horizonte", "irlanda", "sao_paulo"]:
            with open(f"relevant_docs/{arquiv}.dat") as arq:
                dic_relevance_docs[arquiv] = set(arq.readline().split(","))

        return dic_relevance_docs

    def count_topn_relevant(self, n, respostas: List[int], doc_relevantes: Set[int]) -> int:
        """
		Calcula a quantidade de documentos relevantes na top n posições da lista lstResposta que é a resposta a uma consulta
		Considere que respostas já é a lista de respostas ordenadas por um método de processamento de consulta (BM25, Modelo vetorial).
		Os documentos relevantes estão no parametro docRelevantes
		"""
        relevance_count = 0
        n_first = respostas[:n]

        for rd in doc_relevantes:
            if rd in n_first:
                relevance_count = relevance_count + 1

        return relevance_count

    def get_query_term_occurence(self, query: str) -> Mapping[str, TermOccurrence]:
        """
			Preprocesse a consulta da mesma forma que foi preprocessado o texto do documento (use a classe Cleaner para isso).
			E transforme a consulta em um dicionario em que a chave é o termo que ocorreu
			e o valor é uma instancia da classe TermOccurrence (feita no trabalho prático passado).
			Coloque o docId como None.
			Caso o termo nao exista no indic, ele será desconsiderado.
		"""
        map_term_occur = {}
        for term in query.split(" "):
            pre_p_word = self.cleaner.preprocess_word(term)
            ocl = self.index.get_occurrence_list(pre_p_word)
            if len(ocl) != 0:
                to = TermOccurrence(None, ocl[0].term_id, 1)
                if pre_p_word in map_term_occur:
                    map_term_occur[pre_p_word].term_freq = to.term_freq + 1
                else:
                    map_term_occur[pre_p_word] = to

        return map_term_occur

    def get_occurrence_list_per_term(self, terms: List) -> Mapping[str, List[TermOccurrence]]:
        """
        Retorna dicionario a lista de ocorrencia no indice de cada termo passado como parametro.
        Caso o termo nao exista, este termo possuirá uma lista vazia
		"""

        dic_terms = {}
        for term in terms:
            dic_terms[term] = self.index.get_occurrence_list(term)

        return dic_terms

    def get_docs_term(self, query: str) -> List[int]:
        """
			A partir do indice, retorna a lista de ids de documentos desta consulta
			usando o modelo especificado pelo atributo ranking_model
		"""
        # Obtenha, para cada termo da consulta, sua ocorrencia por meio do método get_query_term_occurence
        dic_query_occur = self.get_query_term_occurence(query)

        # obtenha a lista de ocorrencia dos termos da consulta
        dic_occur_per_term_query = self.get_occurrence_list_per_term(list(dic_query_occur.keys()))

        pesos = self.ranking_model.get_ordered_docs(dic_query_occur, dic_occur_per_term_query)
        return pesos

    @staticmethod
    def runQuery(query: str, index: Index, precomp: IndexPreComputedVals, cleaner: Cleaner):
        relevant_doc = read("Insira uma das opções: belo_horizonte irlanda sao_paulo")

        time_checker = CheckTime()
        time_checker.print_delta("Query Creation")

        # PEça para usuario selecionar entre Booleano ou modelo vetorial para intanciar o QueryRunner
        # apropriadamente. NO caso do booleano, vc deve pedir ao usuario se será um "and" ou "or" entre os termos.
        # abaixo, existem exemplos fixos.
        rank_model_choose = read("VectorRankingModel (1) - BooleanRankingModel AND (2) - BooleanRankingModel OR (3)")
        rank_model = None
        if rank_model_choose == 1:
            rank_model = VectorRankingModel(precomp)
        elif rank_model_choose == 2:
            rank_model = BooleanRankingModel(OPERATOR.AND)
        else:
            rank_model = BooleanRankingModel(OPERATOR.OR)

        qr = QueryRunner(rank_model, index, cleaner)
        map_relevantes = qr.get_relevance_per_query()[relevant_doc]

        # Utilize o método get_docs_term para obter a lista de documentos que responde esta consulta
        doc_ids, weights = qr.get_docs_term(query)
        time_checker.print_delta("anwered with {len(respostas)} docs")

        # nesse if, vc irá verificar se o termo possui documentos relevantes associados a ele
        # se possuir, vc deverá calcular a Precisao e revocação nos top 5, 10, 20, 50.
        # O for que fiz abaixo é só uma sugestao e o metododo countTopNRelevants podera
        # auxiliar no calculo da revocacao e precisao
        if len(doc_ids) > 0:
            arr_top = [5, 10, 20, 50]
            for n in arr_top:
                n_tops = qr.count_topn_relevant(n, doc_ids, map_relevantes)
                revocacao = len(doc_ids) - n_tops
                precisao = n_tops
                print(f"Precisao @{n}: {precisao}")
                print(f"Recall @{n}: {revocacao}")

    @staticmethod
    def main():
        print("Starting...")

        index = FileIndex()

        # Create cleaner
        print("Creating cleaner...")
        cleaner = Cleaner(stop_words_file="stopwords.txt", language="portuguese",
                          perform_stop_words_removal=False, perform_accents_removal=False,
                          perform_stemming=False)

        # Checagem se existe um documento (apenas para teste, deveria existir)
        print(f"Existe o doc? index.hasDocId(105047)")

        # Instancie o IndicePreCompModelo para precomputar os valores necessarios para a query
        print("Precomputando valores atraves do indice...")
        precomp = IndexPreComputedVals(index)
        check_time = CheckTime()
        check_time.print_delta("Precomputou valores")


        # inserir while
        query = read("Insira a query: ex (vocês estejam bem)")
        print("Fazendo query...")
        QueryRunner.runQuery(query, index, precomp, cleaner)
