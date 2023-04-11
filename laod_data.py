import requests
import os
import pandas as pd

import warnings
warnings.filterwarnings('ignore')

url = "http://dataservices.imf.org/REST/SDMX_JSON.svc/"
key = "CompactData/IFS/M.DZ.PCPI_IX"  # adjust codes here

# Navigate to series in API-returned JSON data
data = requests.get(f"{url}{key}").json()["CompactData"]["DataSet"]["Series"]

# clean the data and save to csv
data_list = [[obs.get("@TIME_PERIOD"), obs.get("@OBS_VALUE")] for obs in data["Obs"]]

df = pd.DataFrame(data_list, columns=["date", "CPI"])

df = df.set_index(pd.to_datetime(df["date"]))["CPI"].astype("float")


# Save cleaned dataframe as a csv file
cwd = os.getcwd()
if not os.path.exists(os.path.join(cwd + "results")):
    os.mkdir(cwd + "\\results")
if not os.path.exists(os.path.join(cwd + "\\results\\data")):
    os.mkdir(cwd + "\\results\\data")
df.to_csv(cwd + "\\results\\data\\DZ_Consumption_price_index.csv", header=True)
