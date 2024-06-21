from Classes.PubMed import PubMed
import os
import pandas as pd
import numpy as np
from tqdm import tqdm

data_folder = "./data"
files = [file for file in os.listdir(data_folder) if file.endswith(".qrels")]

data_list = []
for file in files:
    df = pd.read_csv(f"{data_folder}/{file}", sep="\s+",
                     header=None, engine='python')
    folder = file.split(".")[0]
    os.makedirs(f"./data/{folder}", exist_ok=True)
    if len(df.columns) != 4:
        print(
            f"Warning: File {file} has {len(df.columns)} columns instead of 4.")
    df.columns = ["CD", 'Misc', 'PMID', 'label']
    data_list.append(df)

    data = pd.concat(data_list, ignore_index=True)

    data["title"] = pd.NA
    data["abstract"] = pd.NA

    cds = data["CD"].unique()
    pubmed = PubMed()

    for cd in tqdm(cds, desc="Processing CDs"):
        cd_df = data[data["CD"] == cd].copy()

        pmids = cd_df['PMID'].astype(str).values
        article_data = pubmed.get_article_data_from_list(pmids)

        for article in article_data:
            cd_df.loc[cd_df["PMID"] == int(
                article.pmid), "title"] = article.title
            cd_df.loc[cd_df["PMID"] == int(
                article.pmid), "abstract"] = article.abstract

        data.loc[data["CD"] == cd, ["title", "abstract"]
                 ] = cd_df[["title", "abstract"]]

        cd_df.to_csv(f"./data/{folder}/{cd}.csv", index=False)
