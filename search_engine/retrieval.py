"""
This module contains the retrieval functions to return the search results
"""
from collections import defaultdict, Counter
import re
from helpers.misc import create_default_dict_list
import numpy as np
from operator import itemgetter
import time
import datetime
import csv


def find_docs_with_term(term, index):
    """
    Returns all doc_ids which contain the "term".

    :param term: The searched term (str)
    :param index: Index in which to search (dict)
    :return: list of relevant doc ids (list)
    """
    if term in index.keys():
        rel_doc_ids = list(index[term].keys())
    else:
        rel_doc_ids = []

    return rel_doc_ids


def get_rel_doc_pos(term, index):
    """
    Returns all relevant doc_positions of a term

    :param term: The searched term (str)
    :param index: Index in which to search (dict)
    :return: positions of term for related doc ids (list of lists)
    """
    if term in index.keys():
        rel_doc_pos = index[term]
    else:
        rel_doc_pos = dict()

    return rel_doc_pos


def get_tfs_docs(term, index):
    """
    Returns term frequencies of a term in the documents

    :param term: The searched term (str)
    :param index: Index in which to search (dict)
    :return: term frequencies of term for related doc ids (dict)
    """
    tfs_docs = index[term].copy()
    tfs_docs = defaultdict(int, tfs_docs)
    for doc in tfs_docs.keys():
        tfs_docs[doc] = len(tfs_docs[doc])

    return tfs_docs


def get_tfs_docs_bool_search(rel_docs, search_results, bool_vals):
    """
    Returns term frequencies of a boolean query

    :param rel_docs: List of all doc_ids which are relevant for this boolean search (list)
    :param search_results: Results of document and tfs for each individual search term (dict)
    :param bool_vals: List of "&&", "&&--" or "||" or "||--" (list)
    :return: term frequencies which are relevant for this boolean search (dict)
    """
    terms = list(search_results.keys())
    # Extract tfs for the first term as basis for summation
    tfs_docs = search_results[terms[0]]["tfs_docs"].copy()

    for idx, bool_val in enumerate(bool_vals):

        if bool_val in ["&&", "&&--", "||"]:
            for doc in rel_docs:
                tfs_docs[doc] = tfs_docs[doc] + search_results[terms[idx + 1]]["tfs_docs"][doc]
        elif bool_val == "||--":
            # Note that here the additional term includes the documents in which the term was found.
            # However, we are interested in the documents in which t2 is not found, hence we add
            # "1" to the tf if it was not found.
            for doc in rel_docs:
                if doc not in search_results[terms[idx + 1]]["tfs_docs"].keys():
                    tf_doc_t_new = 0.5
                else:
                    tf_doc_t_new = 0
                tfs_docs[doc] = tfs_docs[doc] + tf_doc_t_new
        else:
            raise Exception(
                "bool_val of search doesn't match either '&&', '&&--', '||' or '||--'. It is: {}.".format(bool_val))

    return tfs_docs


def bool_search(search_results, indexer, bool_vals):
    """
    Executes a boolean search between relevant documents for all searched terms.

    :param search_results: Results of document and tfs for each individual search term (dict)
    :param indexer: Class instance for the created index (Indexer)
    :param bool_vals: List of "&&", "&&--" or "||" or "||--" (list)
    :return: List of all doc_ids which are relevant for this boolean search (list)
    """
    terms = list(search_results.keys())
    # Extract relevant documents for the first term as basis for boolean analysis
    rel_docs = search_results[terms[0]]["rel_docs"].copy()

    for idx, bool_val in enumerate(bool_vals):
        if bool_val == "&&":
            rel_docs = [doc_id for doc_id in rel_docs if doc_id in search_results[terms[idx + 1]]["rel_docs"]]

        elif bool_val == "&&--":
            rel_docs = [doc_id for doc_id in rel_docs if doc_id not in search_results[terms[idx + 1]]["rel_docs"]]

        elif bool_val == "||":
            rel_docs = list(set(rel_docs + search_results[terms[idx + 1]]["rel_docs"]))

        elif bool_val == "||--":
            rel_docs = list(set([doc_id for doc_id in indexer.all_doc_ids if doc_id not in
                                 search_results[terms[idx + 1]]["rel_docs"]] + rel_docs))

        else:
            raise Exception(
                "bool_val of search doesn't match either '&&', '&&--', '||' or '||--'. It is: {}.".format(
                    bool_val))

    return rel_docs


def simple_proximity_search(search_results, indexer, n=1, pos_asterisk=None, phrase=False):
    """
    Calculates if terms in query are in the same document with less or equal to n distance and return relevant doc_ids.
    Option to perform phrase search.

    :param search_results: Results of document and tfs for each individual search term (dict)
    :param indexer: Class instance for the created index (Indexer)
    :param n: allowed distance in one document (int)
    :param pos_asterisk: positions where the asterisks are located (list)
    :param phrase: whether or not the search is a phrase search and ordering matters (bool)
    :return: List of all doc_ids which are relevant for proximity search (list)
    """
    # Performs a boolean search to get documents which contain both terms
    terms = list(search_results.keys())
    rel_documents_all_terms = bool_search(search_results, indexer=indexer, bool_vals=["&&"] * (len(terms) - 1))

    # Todo: Think about if proximity search considers the max distance between first and last
    # if any(|pos_1 - pos_2|<= n) --> doc_id is relevant --> append it to returned final_rel_doc_ids list
    # Find potential candidates (differentiation important for multiple words in phrase/proximity search)
    final_rel_doc_ids = list()
    for doc_id in rel_documents_all_terms:
        dict_candi = defaultdict(list)
        for idx, term in enumerate(terms[:-1]):
            for pos_1 in search_results[terms[idx]]["rel_doc_pos"][doc_id]:
                for pos_2 in search_results[terms[idx+1]]["rel_doc_pos"][doc_id]:
                    if phrase:
                        diff = 1
                        if pos_asterisk is not None:
                            if (idx + 1) in pos_asterisk:
                                diff = 1 + len([num for num in pos_asterisk if num == idx + 1])
                        if int(pos_2) - int(pos_1) == diff:
                            dict_candi[idx].append((pos_1, pos_2))
                    else:
                        if abs(int(pos_1) - int(pos_2)) <= n:
                            dict_candi[idx].append((pos_1, pos_2))

        # Check if found candidates are actual true positives
        if len(terms) <= 2:
            # If only two terms were compared, take all results
            for _ in dict_candi[0]:
                final_rel_doc_ids.append(doc_id)
        else:
            # For multiple terms delete all positions which are true across terms
            # Finally take the minimum true positions per term as the number for which the whole
            # phrase or proximity was found in the document
            for idx in np.arange(len(terms[:-1])-1):
                for idx2, candi_1 in enumerate(dict_candi[idx]):
                    candi_2 = [pos[0] for pos in dict_candi[idx+1]]
                    if candi_1[1] not in candi_2:
                        del dict_candi[idx][idx2]
            max_cand = min([len(dict_candi[key]) for key in dict_candi.keys()])
            for i in np.arange(max_cand):
                final_rel_doc_ids.append(doc_id)

    # Convert results to appropriate output format
    tfs_docs = dict(Counter(final_rel_doc_ids))
    tfs_docs = defaultdict(int, tfs_docs)
    final_rel_doc_ids = sorted(list(set(final_rel_doc_ids)))

    return final_rel_doc_ids, tfs_docs


def simple_tfidf_search(terms, indexer):
    """
    Calculates the TF-IDF score for multiple terms and returns an ordered dict.

    :param terms: List of terms contained in search (list)
    :param indexer: Class instance for the created index (Indexer)
    :return: Descending sorted pseudo-dictionary with doc_id as key and TF-IDF as value (list)
    """
    doc_relevance = {}
    total_num_docs = len(indexer.all_doc_ids)

    for t in terms:
        rel_docs = find_docs_with_term(t, indexer.index)
        df = len(rel_docs)
        rel_doc_pos = get_rel_doc_pos(t, indexer.index)
        tfs_docs = [len(rel_doc_pos[key]) for key in rel_doc_pos]
        weights_docs = [(1 + np.log10(tf)) * np.log10(total_num_docs / df) for tf in tfs_docs]

        for doc_id, weight in zip(rel_docs, weights_docs):

            if doc_id not in doc_relevance:
                doc_relevance[doc_id] = weight
            else:
                doc_relevance[doc_id] += weight

    sorted_relevance = sorted(doc_relevance.items(), key=lambda x: x[1], reverse=True)
    return sorted_relevance


def calculate_tfidf(rel_docs, tfs_docs, indexer):
    """
    Calculates the TF-IDF score for given search results for one search term

    :param rel_docs: List of relevant documents (list)
    :param tfs_docs: Term frequency in the relevant documents (list)
    :param indexer: Class instance for the created index (Indexer)
    :return: Descending sorted dictionary with doc_id as key and TF-IDF as value (dict)
    """
    doc_relevance = {}
    total_num_docs = len(indexer.all_doc_ids)
    df = len(rel_docs)  # document frequency

    # Calculate the weights per document
    if df > 0:
        # Please note that the calculation was adjusted to differentiate documents in ranking even if
        # all documents of the collection are part of the relevant documents
        if total_num_docs == df:
            weights_docs = [(1 + np.log10(tfs_docs[key])) * 1 for key in rel_docs]
        else:
            weights_docs = [(1 + np.log10(tfs_docs[key])) * (np.log10(total_num_docs / df)) for key in rel_docs]
    else:
        weights_docs = []

    for doc_id, weight in zip(rel_docs, weights_docs):
        if doc_id not in doc_relevance:
            doc_relevance[doc_id] = weight
        else:
            doc_relevance[doc_id] += weight

    sorted_relevance = sorted(doc_relevance.items(), key=lambda x: x[1], reverse=True)

    return sorted_relevance


def execute_search(query, indexer, preprocessor):
    """
    Checks, which type of search has to be done.
    Executes the search and returns the resulting, relevant doc_ids and tfs for those

    :param query: The query which should be searched for (str)
    :param indexer: Class instance for the created index (Indexer)
    :param preprocessor: Preprocessor class instance (Preprocessor)
    :return: List from the matching function containing all relevant doc_ids, List of all tfs for relevant doc_ids
    """
    # compile search patterns to test for
    bool_pattern = re.compile(r"(&&--)|(\|\|--)|(&&)|(\|\|)")
    prox_pattern = re.compile(r"#\d+")
    phra_pattern = re.compile(r'^".*"$')

    # check if boolean search
    if bool_pattern.search(query) is not None:

        type_of_bool_search = bool_pattern.findall(query)
        for idx, item in enumerate(type_of_bool_search):
            type_of_bool_search[idx] = [x for x in item if x != ""][0].strip()

        terms = list()
        pos_bool_old = 0
        for idx, bool in enumerate(type_of_bool_search):
            pos_bool_new = query[pos_bool_old:].find(type_of_bool_search[idx]) + pos_bool_old
            terms.append(query[pos_bool_old:pos_bool_new].strip())
            pos_bool_old = pos_bool_new + len(bool)
        terms.append(query[pos_bool_old:].strip())

        search_results = defaultdict(create_default_dict_list)
        for term in terms:
            search_results[term]["rel_docs"], search_results[term]["tfs_docs"] = execute_search(term, indexer,
                                                                                                preprocessor)

        rel_docs = bool_search(search_results, indexer=indexer, bool_vals=type_of_bool_search)
        tfs_docs = get_tfs_docs_bool_search(rel_docs, search_results, bool_vals=type_of_bool_search)

        return rel_docs, tfs_docs

    # check if proximity search
    elif prox_pattern.search(query) is not None:
        n = int(prox_pattern.search(query).group(0)[1:])
        terms = [re.sub('[^a-zA-Z]+', '', term) for term in query.split(",")]

        search_results = defaultdict(create_default_dict_list)
        for term in terms:
            search_results[term]["rel_docs"], _ = execute_search(term, indexer, preprocessor)
            search_results[term]["rel_doc_pos"] = get_rel_doc_pos(preprocessor.preprocess(term)[0], indexer.index)

        rel_docs, tfs_docs = simple_proximity_search(search_results, indexer=indexer, n=n)

        return rel_docs, tfs_docs

    # check if phrase search --> same as proximity search with n = 1
    elif phra_pattern.search(query) is not None:
        terms = re.sub('"', "", query).split()

        # Check if asterisk is in query
        if "*" in terms:
            pos_asterisk = list()
            for idx, term in enumerate(terms):
                if term == "*":
                        if len(pos_asterisk) > 0:
                            if pos_asterisk[-1] == (idx -1):
                                pos_asterisk.append(idx - 1)
                            else:
                                pos_asterisk.append(idx - len(pos_asterisk))
                        else:
                            pos_asterisk.append(idx - len(pos_asterisk))

            terms = [term for term in terms if term != "*"]
        else:
            pos_asterisk = None

        search_results = defaultdict(create_default_dict_list)
        for term in terms:
            search_results[term]["rel_docs"], _ = execute_search(term, indexer, preprocessor)
            search_results[term]["rel_doc_pos"] = get_rel_doc_pos(preprocessor.preprocess(term)[0], indexer.index)

        if len(search_results.keys()) < 2:
            key = list(search_results.keys())[0]
            final_rel_doc_ids = search_results[key]["rel_docs"]

            # Convert results to appropriate output format
            tfs_docs = dict(Counter(final_rel_doc_ids))
            tfs_docs = defaultdict(int, tfs_docs)
            final_rel_doc_ids = sorted(list(set(final_rel_doc_ids)))
            return final_rel_doc_ids, tfs_docs
        else:
            rel_docs, tfs_docs = simple_proximity_search(search_results, indexer=indexer, n=1, phrase=True, pos_asterisk = pos_asterisk)
            return rel_docs, tfs_docs

    # if nothing else matches --> make a simple search
    else:
        results = find_docs_with_term(preprocessor.preprocess(query)[0], indexer.index)
        tfs_docs = get_tfs_docs(preprocessor.preprocess(query)[0], indexer.index)

        return results, tfs_docs


def execute_queries_and_save_results(query, search_type, indexer, preprocessor, config):
    """
    Function to execute search and return results
    :param query: Query that should be searched (str)
    :param search_type: Parameter which search type it is (str)
    :param indexer: Class instance for the created index (Indexer)
    :param preprocessor: Preprocessor class instance (Preprocessor)
    :param config: Defined configuration settings (dict)
    :return: results (list)
    """

    if search_type == "boolean_and_tfidf": # Todo: Take if clause out once not needed anymore for testing

        if preprocessor.replacement_patterns:
            query = preprocessor.replace_replacement_patterns(query)

        # Execute search for boolean queries considering ranking
        boolean_search_pattern = re.compile('(&&--)|(\|\|--)|(&&)|(\|\|)|(#\d+)|^".*"$')

        # check if boolean search component is in query, then execute boolean search
        # else execute tfidf search
        if boolean_search_pattern.search(query) is not None:
            rel_docs, tfs_docs = execute_search(query, indexer, preprocessor)
            rel_docs_with_tfidf = calculate_tfidf(rel_docs, tfs_docs, indexer)
        else:
            terms = query.split()
            terms = [preprocessor.preprocess(term)[0] for term in terms if len(preprocessor.preprocess(term)) > 0]
            rel_docs_with_tfidf = simple_tfidf_search(terms, indexer)

        if len(rel_docs_with_tfidf) > 0:
            # Only keep top results
            if len(rel_docs_with_tfidf) > config["retrieval"]["number_ranked_documents"]:
                rel_docs_with_tfidf = rel_docs_with_tfidf[:config["retrieval"]["number_ranked_documents"]]

            if config["retrieval"]["result_checking"]:
                # Rescale the results. For queries with "OR NOT" it can happen that the difference in scores between the
                # documents are very low (0.0001). To interpret results easier we re-scale here based on the highest score
                max_value = max(rel_docs_with_tfidf, key=itemgetter(1))[1]
                rel_docs_with_tfidf_scaled = list()
                for idx, _ in enumerate(rel_docs_with_tfidf):
                    rel_docs_with_tfidf_scaled.append((rel_docs_with_tfidf[idx][0], rel_docs_with_tfidf[idx][1] / max_value * 10))

                # Save results on local disk
                ts = time.time()
                st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d_%H_%M_%S')

                with open(st + "_search_results.csv", 'w', newline="") as out:
                    csv_out = csv.writer(out)
                    csv_out.writerow(['doc_id', 'score'])
                    csv_out.writerows(rel_docs_with_tfidf_scaled)

            else:
                rel_docs_with_tfidf_scaled = rel_docs_with_tfidf

            # Write output (only document ids), implementation efficient as maximal number of results 10-100
            results = []
            for doc_id, value in rel_docs_with_tfidf_scaled:
                results.append((doc_id,round(value, 4)))
            return results
