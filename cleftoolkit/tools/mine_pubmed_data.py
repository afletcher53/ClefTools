import os
import pandas as pd
from tqdm import tqdm
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from cleftoolkit.Classes.PubMed import PubMed

def setup_logging():
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')

def get_qrels_files(data_folder):
    return [file for file in os.listdir(data_folder) if file.endswith(".qrels")]

def read_qrels_file(file_path):
    df = pd.read_csv(file_path, sep="\s+", header=None, engine='python')
    if len(df.columns) != 4:
        logging.warning(f"Warning: File {file_path} has {len(df.columns)} columns instead of 4.")
    df.columns = ["CD", 'Misc', 'PMID', 'label']
    return df

def process_cd(cd, cd_df, pubmed, folder):
    pmids = cd_df['PMID'].astype(str).values
    try:
        article_data = pubmed.get_article_data_from_list(pmids)
        for article in article_data:
            cd_df.loc[cd_df["PMID"] == int(article.pmid), "title"] = article.title
            cd_df.loc[cd_df["PMID"] == int(article.pmid), "abstract"] = article.abstract
        cd_df.to_csv(f"./data/{folder}/{cd}.csv", index=False)
        return cd, cd_df
    except Exception as e:
        logging.error(f"Error processing CD {cd}: {str(e)}")
        return cd, None

def main():
    setup_logging()
    data_folder = "./data"
    files = get_qrels_files(data_folder)
    
    data = pd.concat([read_qrels_file(f"{data_folder}/{file}") for file in files], ignore_index=True)
    data["title"] = pd.NA
    data["abstract"] = pd.NA
    
    cds = data["CD"].unique()
    pubmed = PubMed()
    
    with ThreadPoolExecutor(max_workers=5) as executor:
        future_to_cd = {executor.submit(process_cd, cd, data[data["CD"] == cd].copy(), pubmed, files[0].split(".")[0]): cd for cd in cds}
        
        for future in tqdm(as_completed(future_to_cd), total=len(cds), desc="Processing CDs"):
            cd, cd_df = future.result()
            if cd_df is not None:
                data.loc[data["CD"] == cd, ["title", "abstract"]] = cd_df[["title", "abstract"]]
    
    data.to_csv(f"./data/combined_data.csv", index=False)
    logging.info("Processing completed successfully.")

if __name__ == "__main__":
    main()