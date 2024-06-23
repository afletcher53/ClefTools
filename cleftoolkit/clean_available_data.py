import os
import pandas as pd

from Classes.PubMed import PubMed


class Cleaner:
    def __init__(self):
        self.data_folder = "./data"
        self.empty_abstract = "Abstract not available"
        self.empty_titles, self.empty_abstracts, self.abstract_not_availables = self._get_empties()

    def _get_empties(self):
        files = self._get_file_names()
        empty_titles = []
        empty_abstracts = []
        abstract_not_available = []
        for file in files:
            data = pd.read_csv(os.path.join(
                self.data_folder, file), sep=",", engine='python')
            empty_titles.extend([
                (file, pmid) for pmid, title in zip(data["PMID"], data["title"])
                if pd.isna(title)
            ])
            empty_abstracts.extend([
                (file, pmid) for pmid, abstract in zip(data["PMID"], data["abstract"])
                if pd.isna(abstract)
            ])
            abstract_not_available.extend([
                (file, pmid) for pmid, abstract in zip(data["PMID"], data["abstract"])
                if abstract == self.empty_abstract
            ])
        return empty_titles, empty_abstracts, abstract_not_available

    def retry_empty_abstracts(self):
        # get files from self.empty_abstracts and pmid
        files, pmid = zip(*self.empty_abstracts)
        pubmed = PubMed()
        for index, file in enumerate(files):
            try:
                article_data = pubmed.get_article_data_from_list([
                                                                 str(pmid[index])])
                print("Article data: ", article_data)
            except Exception as e:
                print(
                    f"Error retrieving data for PMID {pmid}: {str(e)}")

    def _get_file_names(self) -> list[str]:
        assert os.path.exists(
            self.data_folder), f"Data folder {self.data_folder} does not exist"
        files = []
        for folder in os.listdir(self.data_folder):
            if os.path.isdir(f"{self.data_folder}/{folder}"):
                files += [f"{folder}/{file}" for file in os.listdir(
                    f"{self.data_folder}/{folder}") if file.endswith(".csv")]
        assert len(files) > 0, "No CSV files found in the data folder"
        return files

    def drop_duplicates(self):
        files = self._get_file_names()
        data = pd.DataFrame()
        for file in files:
            data = pd.read_csv(f"{self.data_folder}/{file}",
                               sep=",", engine='python')
            duplicates = data.duplicated(subset="PMID")
            data = data[~duplicates]

            if duplicates.sum() > 0:
                data.to_csv(f"{self.data_folder}/{file}", index=False)

    def get_stats(self):
        files = self._get_file_names()
        print(f"Number of files: {len(files)}")
        print(f"Number of empty titles: {len(self.empty_titles)}")
        print(f"Number of empty abstracts: {len(self.empty_abstracts)}")
        print(
            f"Number of abstracts not available: {len(self.abstract_not_availables)}")


cleaner = Cleaner()
cleaner.get_stats()
cleaner.retry_empty_abstracts()
