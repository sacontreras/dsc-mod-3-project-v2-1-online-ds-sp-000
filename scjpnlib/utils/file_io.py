from ..utils import * 
import os
from pathlib import Path
import urllib
from tqdm import tqdm
import numpy as np
import json
from json_minify import json_minify
import io
from functools import singledispatch

class FileManager:
    def __init__(self):
        pass

    # referenced/used part of https://stackoverflow.com/a/22776
    def download(self, url, local_fname):
        http_file = urllib.request.urlopen(url)
        if http_file.getcode() != 200:
            raise ValueError(f"ERROR {http_file.getcode()} while opening {url}.")

        meta = http_file.info()
        file_size = int(meta['Content-Length'])
        display(HTML(f"Downloading {url} (filesize: {file_size} bytes) to {local_fname}:"))
        file_size_dl = 0
        block_sz = 8192

        fblocks = np.arange(start=0, stop=file_size+1, step=block_sz)
        tqdm_pb = tqdm(fblocks)
        f = open(local_fname, 'wb')
        for tfblock in tqdm_pb: 
            buffer = http_file.read(block_sz)
            if not buffer:
                break
            file_size_dl += len(buffer)
            f.write(buffer)
        f.close()

        file_size_local = os.path.getsize(f.name)
        if file_size_local != file_size:
            display(HTML(f"<h3>Warning!!!  URL file {url} is {file_size} bytes but we only downloaded {file_size_local} bytes to local file {local_fname}!</h3>"))
            if yes_no_prompt(f"Do you want to keep it anyway?  Press 'y' and ENTER to keep incorrectly sized local file {local_fname}, or press any other character and ENTER to delete it."):
                os.remove(local_fname)
                return -1
        else:
            display(HTML(f"<h3>Successfully downloaded {file_size_local}/{file_size} bytes from URL file {url} to local file {local_fname}!</h3>"))
        return 0

    def validate_download(self, fsrcmap):
        for local_file, url in fsrcmap.items():
            while not Path(local_file).is_file():
                if yes_no_prompt(f"File {local_file} is required and does not yet exist locally.  Press 'y' and ENTER to download it now from {url}."):
                    # download and show progress bar
                    if self.download(url, local_file) != 0:
                        break
            display(HTML(f"Required download-file {local_file} exists locally."))

    def load_json(self, f_name):
        with open(f_name, 'r') as f:
            _raw = f.read()
        json_object = json.loads(json_minify(_raw)) # minify is used so that we can place comments/documentation in the JSON config file (which is normally invalid in JSON)
        f.close()
        return json_object


    # apparently np.float32 is not serializable!
    def default(self, o):
        if isinstance(o, np.float32): return np.float64(o)  
        raise TypeError

    def save_json(self, d, fname):
        with io.open(fname, 'w', encoding='utf-8') as f:
            f.write(json.dumps(d, ensure_ascii=False, sort_keys=False, indent=4, default=self.default))
        f.close()

    def append_text_file(self, line, fname):
        with open(fname, 'a') as f:
            f.write(line)
        f.close()