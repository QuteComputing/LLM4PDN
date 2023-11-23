from termcolor import colored
import pandas as pd
import os
import re
import json
from thefuzz import fuzz, process


class DatabaseQuery:
    def __init__(self, csv_database):
        self.csv_database = csv_database
        pd.set_option("display.max_colwidth", None)

    def _match_identifiers(self, query, column, strategy, threshold=None):
        if strategy == "token_sort":
            scores = self.csv_database[column].apply(
                lambda x: fuzz.token_sort_ratio(query, x)
            )
        else:
            scores = self.csv_database[column].apply(lambda x: fuzz.ratio(query, x))

        if threshold is None:
            max_score = scores.max()
            matched_identifiers = self.csv_database[scores == max_score][
                "Identifier"
            ].to_list()
        else:
            matched_identifiers = self.csv_database[scores >= threshold][
                "Identifier"
            ].to_list()

        return matched_identifiers

    def query(self, query, column, strategy="token_sort", threshold=None):
        matched_identifiers = self._match_identifiers(
            query, column, strategy, threshold
        )
        results = self.csv_database[
            self.csv_database["Identifier"].isin(matched_identifiers)
        ].to_dict(orient="records")
        return results

    def query_dataset_by_name(self, dataset_name, threshold=None):
        return self.query(dataset_name, "Dataset Name", "token_sort", threshold)

    def query_dataset_by_location(self, location, threshold=None):
        return self.query(location, "Location", "token_sort", threshold)

    def query_dataset_by_task(self, task, threshold=None):
        return self.query(task, "Task", "token_sort", threshold)

    def query_dataset_used_in_paper(self, paper_name, threshold=None, method="ratio"):
        if method == "token_sort":
            scores = self.csv_database["Research Paper"].apply(
                lambda x: fuzz.token_sort_ratio(paper_name, x)
            )
        else:
            scores = self.csv_database["Research Paper"].apply(
                lambda x: fuzz.ratio(paper_name, x)
            )

        if threshold is None:
            threshold = scores.max()

        matched_papers = self.csv_database[scores >= threshold].to_dict(
            orient="records"
        )
        return matched_papers


def display_results(results):
    for result in results:
        print(colored("-" * 150, "cyan"))
        for key, value in result.items():
            print(colored(f"{key}: {value}", "green"))
    print(colored(f"A total of {len(results)} results are found", "cyan"))


def main_menu(database_query):
    while True:
        print("=" * 150)
        print(colored(f"Which type of query do you want to perform?", "cyan"))
        options = [
            "Query datasets used in a paper",
            "Query papers that used a dataset",
            "Query datasets by location",
            "Query datasets for specific task",
            "Quit",
        ]
        for idx, option in enumerate(options, 1):
            print(colored(f"{idx}. {option}", "white"))

        choice = input("Please input your choice: ")

        if choice == "1":
            paper_name = input("Please input the paper name: ")
            results = database_query.query_dataset_used_in_paper(paper_name)
            display_results(results)
        elif choice == "2":
            dataset_name = input("Please input the dataset name: ")
            results = database_query.query_dataset_by_name(dataset_name, threshold=80)
            display_results(results)
        elif choice == "3":
            location = input("Please input the location: ")
            results = database_query.query_dataset_by_location(location, threshold=85)
            display_results(results)
        elif choice == "4":
            task = input("Please input the task: ")
            results = database_query.query_dataset_by_task(task, threshold=90)
            display_results(results)
        elif choice == "5":
            break


if __name__ == "__main__":
    df = pd.read_csv("../data/paper_dataset_network_cs_8k.csv")
    print(df.columns)
    print(df["Research Paper"].sample(10))

    database_query = DatabaseQuery(df)
    main_menu(database_query)
