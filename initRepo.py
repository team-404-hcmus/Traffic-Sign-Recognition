from concurrent.futures import thread
import requests
import threading
import zipfile

signname_url = f"https://raw.githubusercontent.com/AvivSham/German-Traffic-Signs-Classification/master/signnames.csv"
dataset_url = f"https://d17h27t6h515a5.cloudfront.net/topher/2017/February/5898cd6f_traffic-signs-data/traffic-signs-data.zip"

path_prefix = "./data/"



def download_data(url):
    local_filename =  path_prefix +  url.split('/')[-1]
    # NOTE the stream=True parameter below
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(local_filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192): 
                # If you have chunk encoded response uncomment if
                # and set chunk_size parameter to None.
                #if chunk: 
                f.write(chunk)


t1 = threading.Thread(target=download_data, args=(dataset_url,))
t2 = threading.Thread(target=download_data, args=(signname_url,))

# starting thread 1
t1.start()
# starting thread 2
t2.start()

# wait until thread 1 is completely executed
t1.join()
# wait until thread 2 is completely executed
t2.join()

# both threads completely executed
print("Done!")