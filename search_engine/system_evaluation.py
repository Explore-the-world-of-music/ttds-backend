"""
This module contains the functions to perform the system evaluation
"""

import numpy as np
import pandas as pd

def get_true_positives(results_systems, results_true, cutoff):
    """
    Determine the true positives for a given IR result and the true results

    :param results_systems: Results given by IR system
    :param results_true: True results
    :param cutoff: Limit how many results for the IR system should be considered
    :return: True positives for given IR result
    """
    results_systems_short = results_systems[results_systems["rank_of_doc"] <= cutoff].drop(columns=["rank_of_doc", "score"])

    # Find relevant documents in the IR system results (by using a common identifier query_doc)
    results_systems_short["query_doc"] = results_systems_short["query_number"].astype(str) + "_" + \
                                         results_systems_short["doc_number"].astype(str)

    results_true_lc = results_true.copy()
    results_true_lc["query_doc"] = results_true_lc["query_number"].astype(str) + "_" + \
                                   results_true_lc["doc_number"].astype(str)

    results_systems_short["relevant_doc"] = np.where(results_systems_short["query_doc"].isin(results_true_lc["query_doc"])
                                                     ,1, 0)

    # Sum over all relevant docs to find true positives
    results_systems_short = results_systems_short.groupby(["query_number"])["relevant_doc"].agg("sum").reset_index()

    return results_systems_short

def calculate_precision(results_systems, results_true, cutoff):
    """
    Function to calculate the precision for a IR system and query

    :param results_systems: Results given by IR system
    :param results_true: True results
    :param cutoff: Limit how many results for the IR system should be considered
    :return: Precision in data frame
    """
    # Derive the true positives
    true_positives = get_true_positives(results_systems, results_true, cutoff)

    # Derive the documents that are retrieved (but not neccessarily relevant)
    results_systems = results_systems[results_systems["rank_of_doc"] <= cutoff]
    retrieved = results_systems.groupby(["query_number"])["doc_number"].agg("count").reset_index(name="n_retrieved")

    # Join TP and retrieved documents to calculate the precision
    df_precision = pd.merge(true_positives, retrieved, how="left", on=["query_number"])
    column_name = "P@" + str(cutoff)
    df_precision[column_name] = df_precision["relevant_doc"] / df_precision["n_retrieved"]

    return df_precision.drop(columns=["relevant_doc", "n_retrieved"])

def calculate_recall(results_systems, results_true, cutoff):
    """
    Function to calculate the recall for a IR system and query

    :param results_systems: Results given by IR system
    :param results_true: True results
    :param cutoff: Limit how many results for the IR system should be considered
    :return: Recall in data frame
    """
    # Calculate the true positives
    true_positives = get_true_positives(results_systems, results_true, cutoff)

    # Derive the relevant documents
    rank_of_doc_col = list()
    results_true_temp = results_true.copy()
    for query in results_true_temp.query_number.unique():
        length = len(results_true_temp.loc[results_true_temp.query_number == query,"query_number"])
        rank_of_doc_col = rank_of_doc_col + list(np.arange(1,length+1))
    results_true_temp["rank_of_doc"] = rank_of_doc_col
    results_true_temp = results_true_temp[results_true_temp["rank_of_doc"] <= cutoff]
    results_true_temp = results_true_temp.drop(columns = ["rank_of_doc"])

    relevant = results_true_temp.groupby(["query_number"])["doc_number"].agg("count").reset_index(name="n_relevant")

    # Join information and calculate recall
    df_recall = pd.merge(true_positives, relevant, how="left", on="query_number")
    df_recall["R@" + str(cutoff)] = df_recall["relevant_doc"] / df_recall["n_relevant"]

    return df_recall.drop(columns=["relevant_doc", "n_relevant"])

def calculate_average_precision(results_systems, results_true):
    """
    Function to calculate the average precision for a IR system and query

    :param results_systems: Results given by IR system
    :param results_true: True results
    :return: Average precision in data frame
    """
    # Initiate new data frames copies and find relevant documents in IR results
    results_systems_ap = results_systems.copy()
    results_true_map = results_true.copy()

    results_systems_ap["query_doc"] = results_systems_ap["query_number"].astype(str) + "_" + \
                                      results_systems_ap["doc_number"].astype(str)
    results_true_map["query_doc"] = results_true_map["query_number"].astype(str) + "_" + \
                                    results_true_map["doc_number"].astype(str)
    results_systems_ap["relevant_doc"] = np.where(results_systems_ap["query_doc"].isin(results_true_map["query_doc"]),
                                                  1, 0)

    # Calculate the relevant documents for each system and query
    relevant_doc_cumsum = results_systems_ap.groupby(["query_number"])["relevant_doc"].cumsum().reset_index(name="cum_rel_doc")
    results_systems_ap = pd.merge(results_systems_ap, relevant_doc_cumsum, how="left", left_index=True,right_index=True)

    # Calculate the AP for each query system and query
    results_systems_ap["AP_ind"] = np.where(results_systems_ap["relevant_doc"] == 1,
                                            results_systems_ap["cum_rel_doc"] / results_systems_ap["rank_of_doc"],
                                            np.nan)

    # Sum over individual Average Precision to get total AP for each query and system
    results_systems_ap_grouped = results_systems_ap.groupby(["query_number"])["AP_ind"].sum().reset_index(name="AP_sum")
    results_systems_ap_grouped["AP_sum"] = results_systems_ap_grouped["AP_sum"].fillna(0)

    # Get relevant documents for each query to divided the IR system results by those
    relevant_docs = results_true_map.groupby("query_number")["doc_number"].count().reset_index(name="n_relevant")
    results_systems_ap_grouped = pd.merge(results_systems_ap_grouped, relevant_docs, how="left", on="query_number")

    # Get final AP number
    results_systems_ap_grouped["AP"] = results_systems_ap_grouped["AP_sum"] / results_systems_ap_grouped["n_relevant"]

    return results_systems_ap_grouped[["query_number", "AP"]]

def calculate_discounted_cumulative_gain(results_systems, results_true, rank):
    """
    Function to calculate the NORMALIZED discounted cumulative gain (nDCG) for a given IR system and query

    :param results_systems: Results given by IR system
    :param results_true: True results
    :param rank: Rank which should be considered for the DCG
    :return: nDCG@k for a given rank k as part of data frame
    """
    # Calculate DCG for the given IR system results
    df_dcg = pd.merge(results_systems, results_true, how="left", on=["query_number", "doc_number"])
    df_dcg["relevance"] = df_dcg["relevance"].fillna(0)
    df_dcg["DG"] = np.where(df_dcg["rank_of_doc"] > 1, df_dcg["relevance"] / np.log2(df_dcg["rank_of_doc"]),
                            df_dcg["relevance"])
    DG_cumsum = df_dcg.groupby(["query_number"])["DG"].cumsum().reset_index(name="DCG@k")
    df_dcg = pd.merge(df_dcg, DG_cumsum, how="left", left_index=True, right_index=True).drop(columns=["index"])

    # Build data frame that represent the ideal DCG ordering
    df_dcg_ideal = results_true.copy()
    df_dcg_ideal["rank_of_doc"] = 1
    rank_of_doc_ideal = df_dcg_ideal.groupby(["query_number"])["rank_of_doc"].cumsum()
    df_dcg_ideal["rank_of_doc"] = rank_of_doc_ideal
    df_dcg_ideal.rename(columns={"relevance": "relevance_ideal"}, inplace=True)
    df_dcg_ideal = df_dcg_ideal.drop(columns=['doc_number'])

    # Join the IR system DCG data frame with the ideal one and calculate key metrics
    df_dcg_combined = pd.merge(df_dcg, df_dcg_ideal, how="left", on=["query_number", "rank_of_doc"])
    df_dcg_combined["relevance_ideal"] = df_dcg_combined["relevance_ideal"].fillna(0)

    df_dcg_combined["iDG"] = np.where(df_dcg_combined["rank_of_doc"] > 1,
                                      df_dcg_combined["relevance_ideal"] / np.log2(df_dcg_combined["rank_of_doc"]),
                                      df_dcg_combined["relevance_ideal"])
    DG_cumsum_ideal = df_dcg_combined.groupby(["query_number"])["iDG"].cumsum().reset_index(name="iDCG@k")
    df_dcg_combined = pd.merge(df_dcg_combined, DG_cumsum_ideal, how="left", left_index=True, right_index=True).drop(
        columns=["index"])

    # Calculate the final nDCG for each rank and then filter on the rank
    df_dcg_combined["nDCG@" + str(rank)] = df_dcg_combined["DCG@k"] / df_dcg_combined["iDCG@k"]

    # Todo: Discuss with Jannis
    nDCG_for_specified_rank = pd.DataFrame()
    for query in df_dcg_combined.query_number.unique():

        max_rank = min(int(rank), max(df_dcg_combined.loc[df_dcg_combined.query_number == query,"rank_of_doc"]))
        nDCG_for_specified_rank = nDCG_for_specified_rank.append(
            df_dcg_combined.loc[df_dcg_combined["rank_of_doc"] == max_rank, ["query_number", "nDCG@" + str(rank)]])
        nDCG_for_specified_rank.reset_index(inplace=True, drop=True)

    return nDCG_for_specified_rank