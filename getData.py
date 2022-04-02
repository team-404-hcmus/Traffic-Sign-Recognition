import os
import requests
import threading
import zipfile
import queue
import concurrent.futures
from os import path


links = [
    f"https://raw.githubusercontent.com/AvivSham/German-Traffic-Signs-Classification/master/signnames.csv",
    f"https://d17h27t6h515a5.cloudfront.net/topher/2017/February/5898cd6f_traffic-signs-data/traffic-signs-data.zip"
]

path_prefix = "./data/"

if not path.exists(path_prefix):
    os.mkdir(path_prefix)


def download_data(url):
    local_filename =  path_prefix +  url.split('/')[-1]
    if path.exists(local_filename):
        print("{} - already exist".format(local_filename))
        return local_filename
    # NOTE the stream=True parameter below
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(local_filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192): 
                f.write(chunk)


with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
    # Start the load operations and mark each future with its URL
    future_to_url = {executor.submit(download_data, link): link for link in links}
    
    def finish_callback(file_name):
        if data.split(".")[-1] == "zip":
            print("Extracting - {}".format(data))
            with zipfile.ZipFile(data, 'r') as zip_ref:
                zip_ref.extractall(path_prefix)

                
    for future in concurrent.futures.as_completed(future_to_url):
            
        try:
            data = future.result()
            future.add_done_callback(finish_callback)
        except Exception as exc:
            pass

print("Done!")