import os
import pandas as pd
from tqdm import tqdm
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import threading
import psutil

from cleftoolkit.Classes.PubMed import PubMed

# Global stop event
stop_event = threading.Event()

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
            if stop_event.is_set():
                return cd, None
            cd_df.loc[cd_df["PMID"] == int(article.pmid), "title"] = article.title
            cd_df.loc[cd_df["PMID"] == int(article.pmid), "abstract"] = article.abstract
        cd_df.to_csv(f"./data/{folder}/{cd}.csv", index=False)
        return cd, cd_df
    except Exception as e:
        logging.error(f"Error processing CD {cd}: {str(e)}")
        return cd, None

class WorkerMonitor:
    def __init__(self, executor):
        self.executor = executor
        self.active_workers = 0
        self.lock = threading.Lock()
        self.current_process = psutil.Process()

    def increment(self):
        with self.lock:
            self.active_workers += 1

    def decrement(self):
        with self.lock:
            self.active_workers -= 1

    def get_active_workers(self):
        with self.lock:
            return self.active_workers

    def get_thread_count(self):
        return self.current_process.num_threads()

    def should_stop(self):
        return stop_event.is_set()

def update_progress_bar(pbar, worker_monitor):
    while not pbar.disable and not stop_event.is_set():
        active_workers = worker_monitor.get_active_workers()
        total_workers = worker_monitor.executor._max_workers
        thread_count = worker_monitor.get_thread_count()
        pbar.set_description(f"Processing CDs (Workers: {active_workers}/{total_workers}, Threads: {thread_count})")
        time.sleep(1)  # Update every second

def process_cd_with_monitor(cd, cd_df, pubmed, folder, worker_monitor):
    worker_monitor.increment()
    try:
        if worker_monitor.should_stop():
            return cd, None
        result = process_cd(cd, cd_df, pubmed, folder)
        return result
    finally:
        worker_monitor.decrement()

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
        worker_monitor = WorkerMonitor(executor)
        
        future_to_cd = {executor.submit(process_cd_with_monitor, cd, data[data["CD"] == cd].copy(), pubmed, files[0].split(".")[0], worker_monitor): cd for cd in cds}
        
        with tqdm(total=len(cds), desc="Processing CDs") as pbar:
            update_thread = threading.Thread(target=update_progress_bar, args=(pbar, worker_monitor), daemon=True)
            update_thread.start()
            
            try:
                for future in as_completed(future_to_cd):
                    if stop_event.is_set():
                        break
                    cd, cd_df = future.result()
                    if cd_df is not None:
                        data.loc[data["CD"] == cd, ["title", "abstract"]] = cd_df[["title", "abstract"]]
                    pbar.update(1)
            except KeyboardInterrupt:
                logging.info("Keyboard interrupt received. Stopping...")
                stop_event.set()
                executor.shutdown(wait=False)
            finally:
                stop_event.set()  # Ensure the update thread stops
    
    if not stop_event.is_set():
        data.to_csv(f"./data/combined_data.csv", index=False)
        logging.info("Processing completed successfully.")
    else:
        logging.info("Processing was interrupted.")

if __name__ == "__main__":
    main()