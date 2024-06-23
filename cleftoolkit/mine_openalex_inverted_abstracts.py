# get every csv in the data folder

import os
import pandas as pd

data_folder = "./data"
assert os.path.exists(data_folder), f"Data folder {data_folder} does not exist"
files = [file for file in os.listdir(data_folder) if file.endswith(".csv")]
assert len(files) > 0, "No CSV files found in the data folder"
data = pd.DataFrame()
for file in files:
    data = data.concat(pd.read_csv(f"{data_folder}/{file}", sep="\s+",
                                   header=None, engine='python'))

data["abstract"] = pd.NA
pmids = data["PMID"].unique()


# issue a api call to get openalex inverted abstract information

abstract = "This is an abstract"
# update the data with the abstract information
# 1 locate all the pmids in the data
# 2 Assign the abstract information to the data
# 3 save the data to a new file

relevant_data = data[data["PMID"].isin(pmids)]
relevant_data["abstract"] = abstract

# merge the data with the updated abstract information
data = data.merge(relevant_data, on="PMID", how="left")

# save the data to a new file (via CD), maintaining the original structure (clef_2017_train, clef_2018_train, etc.)
'
