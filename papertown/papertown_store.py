import os
import time
import argparse
from papertown import DatasetStore, DataComposer, load_tokenizer
from tqdm import tqdm

from .papertown_utils import *

def _tobool(s):
    return s.lower() == 'true'

def setup_store():
    parser = argparse.ArgumentParser(description="papertown_store")
    parser.add_argument("files", type=str, nargs="+", help="files")
    parser.add_argument("--tokenizer_path", default=DEFAULT_TOKENIZER)
    parser.add_argument("--store_path", default="store")
    parser.add_argument("--block_size", type=int, default=2048)
    parser.add_argument("--format", default="simple")
    parser.add_argument("--split", default="train")
    parser.add_argument("--padding", type=_tobool, default=True)
    parser.add_argument("--sep", type=str, default=None)
    parser.add_argument("--N", type=int, default=None)
    parser.add_argument("--num_works", type=int, default=0)
    hparams = parser.parse_args()  # hparams になる
    return hparams

def main_store():
    hparams = setup_store()
    tokenizer = load_tokenizer(hparams.tokenizer_path)
    store = DatasetStore(tokenizer=tokenizer, 
                         block_size=hparams.block_size, 
                         dir=hparams.store_path)
    store.upload(filename=hparams.files[0], 
                 format=hparams.format, split=hparams.split, padding=hparams.padding, sep=hparams.sep, N=hparams.N)



def setup_testdata():
    parser = argparse.ArgumentParser(description="papertown_testdata")
    parser.add_argument("--url_list", type=str)
    parser.add_argument("--block_size", type=int, default=1024)
    parser.add_argument("--N", type=int, default=0)
    parser.add_argument("--format", default="")
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    hparams = parser.parse_args()  # hparams になる
    return hparams

def main_testdata():
    hparams = setup_testdata()
    with DataComposer(url_list=hparams.url_list, 
                      split=hparams.format+hparams.split,
                      block_size=hparams.block_size) as dc:
        print(len(dc))
        if hparams.N == 0:
            return
        if hparams.N == -1:
            hparams.N = len(dc)
        start = time.time()
        for index in tqdm(range(hparams.N), total=hparams.N):
            dc[index]
        end = time.time()
        print(f'Total: {end-start:.1f}s Iterations: {hparams.N:,} {hparams.N/(end-start)}[it/s]')


"""
def setup_dump():
    parser = argparse.ArgumentParser(description="papertown_dump")
    parser.add_argument("--tokenizer_path", default=DEFAULT_TOKENIZER)
    parser.add_argument("--url_list", type=str)
    parser.add_argument("--block_size", type=int, default=1024)
    parser.add_argument("--N", type=int, default=0)
    parser.add_argument("--format", default="")
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    hparams = parser.parse_args()  # hparams になる
    return hparams


def main_dump():
    parser.add_argument("files", type=str, nargs="+", help="files")
    parser.add_argument("--tokenizer_path", default=DEFAULT_TOKENIZER)
    tokenizer = load_tokenizer('')
    with DataComposer(url_list=urls, block_size=256) as dc:
        print('len(dc):', len(dc))
        for i in range(len(dc)):
                x = tokenizer.decode(dc[i])
                if i%100==0:
                    print(i, dc[i], x)
"""


if __name__ == "__main__":  # わかります
    main_testdata()

