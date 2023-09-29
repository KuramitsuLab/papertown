import os
import time
import random
from pathlib import Path

import json
import shutil
import subprocess
from typing import List

from collections import deque

from filelock import FileLock
import numpy as np
import gzip

import torch
from transformers import AutoTokenizer
from torch.utils.data import Dataset

from .papertown_utils import *
from .papertown_tokenizer import *
from .splitter import PretrainedTextSplitter, TextSplitter, TextPairSplitter, file_iterator

_ID = 0
def random_name():
    global _ID
    _ID+= 1
    return f'Cache{_ID-1}'

# 設定ファイル

DEFAULT_BLOCK_SIZE = 2048
DEFAULT_MAX_LENGTH = 4096
N_CHUNKS = 4096

# ファイルシステム

def join_path(dir, file):
    if file is None: 
        return dir
    if dir.endswith('/'):
        dir = dir[:-1]
    if file.startswith('/'):
        file = file[1:]
    return f'{dir}/{file}'

def _remove_heading_nL(s):
    while s.startswith('<nL>'):
        s = s[4:]
    return s

def _makedirs(path):
    dir, _,  file = path.rpartition("/")
    if '.' in file: #拡張子が含まれる場合
        os.makedirs(dir,  exist_ok=True)
    elif not os.path.isfile(path):
        os.makedirs(path,  exist_ok=True)

def get_file_sha1(filepath: str):
    # ファイルをバイナリモードで読み込む
    with open(filepath, 'rb') as f:
        # ファイルの内容を読み込む
        content = f.read()
        # SHA-1ハッシュオブジェクトを作成
        sha1 = hashlib.sha1()
        # ファイルの内容をハッシュオブジェクトに追加
        sha1.update(content)
        # 16進数でハッシュ値を取得
        sha1_hexdigest = sha1.hexdigest()
    return sha1_hexdigest

def get_filesize(file_path):
    if os.path.exists(file_path) and os.path.isfile(file_path):
        return os.path.getsize(file_path)
    else:
        return -1

def touch(file_path):
    file = Path(file_path)
    file.touch(exist_ok=True)

def wait_for_file(file_path, timeout=60):
    """
    指定されたファイルが存在するかを定期的にチェックし、
    タイムアウトまでにファイルが見つかった場合は True を返します。
    タイムアウトした場合は False を返します。
    """
    start_time = time.time()
    end_time = start_time + timeout
    while time.time() < end_time:
        if get_filesize(file_path) > 0:
            verbose_print(f'{time.time()-start_time} 秒, 待ちました')
            return True  # ファイルが見つかった
        time.sleep(1)  # 1秒待つ
    return False  # タイムアウト

def resolve_file(url_base, file_path, cache_dir, sync=True):
    remote_file = safe_join(url_base, file_path)
    if remote_file.startswith('/'):
        # ローカルなファイルパスの場合
        return remote_file
    cache_file = safe_join(cache_dir, file_path)
    # ディレクトリを作っておく
    os.makedirs(cache_file.rpartition("/")[0], exist_ok=True)
    cache_file_size = get_filesize(cache_file)
    #print('@', cache_file_size, cache_file)
    if cache_file_size > 0:
        return cache_file

    # ダウンロードコマンド
    if remote_file.startswith('file:'):
        remote_file = os.path.abspath(remote_file[5:]) # file: をとる
        cmd = f'cp {remote_file} {cache_file}'
    else:
        cmd = f"wget -qO {cache_file}.tmp {remote_file} && mv {cache_file}.tmp {cache_file}"

    if sync:
        if cache_file_size == 0:
            verbose_print('ダウンロード中 最大30秒待ちます.', remote_file)
            if wait_for_file(cache_file, 30):
                return cache_file
        touch(cache_file)
        subprocess.call(cmd, shell=True)
        verbose_print(f'Downloaded {get_filesize(cache_file):,} bytes:', cmd)
        return cache_file

    if get_filesize(cache_file) == -1:
        touch(cache_file)
        verbose_print('プレフェッチ', remote_file)
        subprocess.call(f"{cmd} &", shell=True, stderr=subprocess.DEVNULL)
    # else:
    #     verbose_print('既にダウンロード中..', remote_file)
    return None

def chunkseq_to_filename(chunkseq:int, prefix:str, file_ext:str):
    dir = f"{(chunkseq//100):04d}"
    return f"{dir}/{prefix}{(chunkseq%100):02d}.{file_ext}"

def save_chunk_file(base_dir:str, chunk_file:str, chunks:List[np.ndarray]):
    filepath = join_path(base_dir, chunk_file)
    _makedirs(filepath)
    if filepath.endswith('.npz'):
        np.savez_compressed(filepath, *chunks)
    return {'filesize': get_filesize(filepath), 
            'sha1': get_file_sha1(filepath)}

def load_chunk_file(base_dir:str, chunk_file:str=None):
    filepath = join_path(base_dir, chunk_file)
    try:
        #if filepath.endswith('.npz'):
        npz = np.load(filepath)
        return [npz[n] for n in npz.files]
    except BaseException as e:
        verbose_print(f'broken chunk file {chunk_file}: {e}')
        return None

def check_chunk_file(base_dir:str, chunk_file:str, checks: dict):
    filepath = join_path(base_dir, chunk_file)
    if 'filesize' in checks:
        if get_filesize(filepath) != checks['filesize']:
            return False
    if 'sha1' in checks:
        if get_file_sha1(filepath) != checks['sha1']:
            return False
    return True

def make_chunk_filelist(base_dir:str, chunk_files:List[str]):
    d = {}
    for chunk_file in chunk_files:
        if not load_chunk_file(base_dir, chunk_file):
            return None
        filepath = safe_join(base_dir, chunk_file)
        checks = {'filesize': get_filesize(filepath), 'sha1': get_file_sha1(filepath)}
        if not check_chunk_file(base_dir, chunk_file, checks):
            verbose_print(f'broken chunk file {chunk_file}')
            return None
        d[chunk_file] = checks
    return d

def shuffle_chunk_files(base_dir:str, chunk_file:str, chunk_file2:str):
    assert chunk_file != chunk_file2
    chunks = load_chunk_file(base_dir, chunk_file)
    chunks2 = load_chunk_file(base_dir, chunk_file2)
    length = len(chunks)
    merged_chunks = chunks+chunks2
    random.shuffle(merged_chunks)
    save_chunk_file(base_dir, chunk_file, merged_chunks[:length])
    save_chunk_file(base_dir, chunk_file, merged_chunks[length:])

"""
empty_tokens = []

def _block_simply(blocks: List[List[int]], tokens: List[int], block_size=DEFAULT_BLOCK_SIZE, fill=empty_tokens):
    # とりあえず、シンプルにブロックを分割する
    for i in range(0, len(tokens) - block_size + 1, block_size):  
        segmented = tokens[i : i + block_size]
        blocks.append(segmented)
    remaining = len(tokens) % block_size
    if remaining == 0: # 最後の分割が揃っていればおしまい
        return fill
    remaining_tokens = tokens[-remaining:] + fill
    while len(remaining_tokens) >= block_size:
        blocks.append(remaining_tokens[:block_size])
        remaining_tokens = remaining_tokens[block_size:]
    return remaining_tokens

def tokenize_block_sep(tokenizer, blocks: List[List[int]], text:str, 
                       block_size=DEFAULT_BLOCK_SIZE, 
                       fill=empty_tokens, sep=DEFAULT_SEP, overlap=0):
    chunks = [tokenizer.convert_tokens_to_ids(tokenizer.tokenize(line)) for line in text.split(sep)]
    chunks[-1] = tokenizer.build_inputs_with_special_tokens(chunks[-1])
    chunk = []
    for ids in chunks:
        prev_length = len(chunk)
        chunk.extend(ids)
        if len(chunk) >= block_size:
            blocks.append(chunk[:block_size])
            if block_size - prev_length < overlap:
                chunk = _block_simply(blocks, ids, block_size)
            else:
                chunk = _block_simply(blocks, chunk[block_size:], block_size)
    if len(chunk) > 4:
        return _block_simply(blocks, chunk, block_size, fill)
    return fill

def tokenize_text(tokenizer, blocks, text, block_size=DEFAULT_BLOCK_SIZE):
    inputs = tokenizer.encode(text)
    if len(inputs) > block_size:
        half_size = block_size // 2
        prefix = inputs[:half_size]
        suffix = inputs[-half_size:]
        prefix[-1] = find_ellipsis_token_id(tokenizer)
        inputs = prefix + suffix
    blocks.append(inputs)
    return empty_tokens

def tokenize_pair(tokenizer, blocks, inputs, labels, block_size=DEFAULT_BLOCK_SIZE):
    inputs = tokenizer.encode(inputs)
    labels = tokenizer.encode(labels)
    if len(labels) > block_size:
        # ラベルの方が大きい場合は諦める
        return empty_tokens
    if len(inputs)+len(labels) > block_size:
        # ラベルは完全に残す
        half_size = (block_size - len(labels)) // 2
        prefix = inputs[:half_size]
        suffix = inputs[-half_size:]
        prefix[-1] = find_ellipsis_token_id(tokenizer)
        inputs = prefix + suffix
    blocks.append(inputs+labels)
    return empty_tokens

def tokenize_line(tokenizer, blocks, line:str, 
                  block_size=DEFAULT_BLOCK_SIZE, fill=empty_tokens,
                  padding=False, jsonl_key='text', sep=DEFAULT_SEP, overlap=0):
    if jsonl_key is not None:
        d = json.loads(line)
        if 'out' in d and 'in' in d:
            return tokenize_pair(tokenizer, blocks, d['in'], d['out'], block_size=block_size)
        if 'inputs' in d and 'labels' in d:
            return tokenize_pair(tokenizer, blocks, d['inputs'], d['labels'], block_size=block_size)
        line = d[jsonl_key]
    else:
        line = line.rstrip()
    if padding:
        return tokenize_text(tokenizer, blocks, line, block_size=block_size)
    else:
        return tokenize_block_sep(tokenizer, blocks, line, block_size, fill, sep, overlap)

def tokenize_file(tokenizer, filename, update_fn=None, 
           block_size=DEFAULT_BLOCK_SIZE, padding=True, overlap=0, N=None, jsonl_key='text', sep=DEFAULT_SEP):
    if N == -1:
        N = get_file_lines(filename)
    if N:
        from tqdm import tqdm
        pbar = tqdm(total=N, desc=filename)
    if '.jsonl' not in filename:
        jsonl_key = None # jsonl でない
    blocks=[]
    fill = empty_tokens
    with zopen(filename) as f:
        line = f.readline()
        c=1
        while line:
            fill = tokenize_line(tokenizer, blocks, line, block_size, fill, jsonl_key=jsonl_key, padding=padding, overlap=overlap, sep=sep)
            if update_fn is not None:
                update_fn(blocks)
                blocks=[]
            line = f.readline()
            if N: 
                pbar.update()
                if c > N: break
            c+=1
    if N:
        pbar.close()
    return blocks
"""

def stat_tokens(counts):
    if len(counts) == 0:
        return {'total': 0}
    data = np.array(counts)
    return {
        'total': int(np.sum(data)),
        'mean': float(np.mean(data)),
        'std': float(np.var(data)) ** 0.5,
        'max': int(np.max(data)),
        '75%': int(np.percentile(data, 75)),
        'median': int(np.median(data)),
        '25%': int(np.percentile(data, 25)),
        'min': int(np.min(data)),
    }

def safe_splitprefix(s):
    s = str(s)
    if not s.endswith('_'):
        return f'{s}_'
    return s

class DatasetStore(object):
    def __init__(self, dir, **kwargs):
        self.dir = safe_dir(dir)
        self.config = dict(kwargs)
        self.tokenizer = self.config.get("tokenizer", None)
        if self.tokenizer is None:
            verbose_print(f'tokenizer is unset and using default {DEFAULT_TOKENIZER}')
            self.tokenizer = load_tokenizer(DEFAULT_TOKENIZER)
        self.config['tokenizer_path'] = str(self.tokenizer.name_or_path)
        self.config['tokenizer'] = get_tokenizer_info(self.tokenizer)
        self.block_size = self.config.get("block_size", DEFAULT_BLOCK_SIZE)
        self.split_prefix = safe_splitprefix(self.config.get('split', DEFAULT_SPLIT))
        self.file_ext = self.config.get("file_ext", "npz")
        self.n_chunks = self.config.get("n_chunks", N_CHUNKS)
        self.shuffle = self.config.get("shuffle", False)
        self.chunkseq = 0
        self.bufs = []
        self.n_items = 0
        self.token_counts = []
        self.chunk_files = []

    def save_config(self):
        config_file = safe_join(self.dir, f'{self.split_prefix}config.json')
        with open(config_file, "w") as w:
            json.dump(self.config, w)
    
    def save(self, save_config=True):
        if len(self.bufs) > 0:
            chunk_file = chunkseq_to_filename(self.chunkseq, self.split_prefix, self.file_ext)
            save_chunk_file(self.dir, chunk_file, self.bufs)
            self.chunk_files.append(chunk_file)
            self.n_items += len(self.bufs)
            if len(self.bufs) == self.n_chunks:
                self.chunkseq += 1
                self.bufs = []
        if save_config:
            self.config.update(dict(
                block_size=self.block_size,
                n_items=self.n_items,
                n_chunks=self.n_chunks,
                chunkseq=self.chunkseq,
            ))
            file_checks = make_chunk_filelist(self.dir, self.chunk_files)
            if file_checks:
                self.config['filechecks'] = file_checks
            self.n_tokens = self.config['n_tokens'] # 正確な値
            self.save_config()

    def append(self, block: List[int]):
        self.bufs.append(np.array(block, dtype=np.int32))
        self.token_counts.append(len(block))
        if len(self.bufs) == self.n_chunks:
            self.save(save_config=False)

    def extend(self, blocks: List[List[int]]):
        for block in blocks:
            self.append(block)    

    def upload(self, filename, format='pre', split='train', N=None, sep=DEFAULT_SEP):
        if format == 'pre':
            splitter = PretrainedTextSplitter(self.tokenizer, block_size=self.block_size, sep=sep)
            split = 'pretrain'
        elif format == 'seq2seq':
            splitter = TextPairSplitter(self.tokenizer, block_size=self.block_size, sep=sep)
            split = f'seq2seq_{split}'
        else:
            splitter = TextSplitter(self.tokenizer, block_size=self.block_size, sep=sep)
            format = ''
        self.split_prefix = safe_splitprefix(split)
        iterator = file_iterator(filename, N=N, sep=sep)
        splitter.split_iter(iterator=iterator, update_fn=self.extend)
        splitter.report(self.config, verbose=True)
        self.config['format'] = format
        self.save()
        verbose_print(f'Tokens: {self.n_tokens:,} Items: {self.n_items:,} Blocks: {self.block_size:,}')


# ChunkedDataset

class _DummyFileLock:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass

def _FileLock(lockfile: str):
    # lockがNoneなら何もしない
    return _DummyFileLock() if lockfile is None else FileLock(lockfile)


import hashlib

def url_to_hash(url):
    return hashlib.md5(url.encode()).hexdigest()

class ChunkedDataset(Dataset):
    def __init__(self, url, split=DEFAULT_SPLIT, block_size=None, lock_file=None, cache_dir=".", shuffle=True, prefetch=1, **kwargs):
        """
        block_size=Noneのときは、再分割しない
        """
        self.url = safe_dir(url)
        self.split_prefix = safe_splitprefix(split)
        self.cache_dir = join_path(cache_dir, url_to_hash(url))
        self.lock_file = lock_file
        self.config = self.load_config(kwargs)
        self.file_ext = self.config.get("file_ext", "npz")
        self.n_tokens = self.config.get('n_tokens', 0)
        self.n_items = self.config.get("n_items", 0)
        self.n_chunks = self.config.get("n_chunks", N_CHUNKS)
        self.tokenizer_path = self.config.get("tokenizer_path", DEFAULT_TOKENIZER)
        if block_size is None or block_size >= self.config.get('block_size', -1):
            self.n_subblocks = 1  # 再分割しない
        else:
            self.block_size = block_size
            self.n_subblocks = self.config['block_size'] // block_size
        #print('DEBUG: n_subblocks', self.n_subblocks, block_size, self.config.get('block_size', -1))
        self.shuffle = shuffle
        self.queue = deque(maxlen=64)
        self.cache = {}
        self.prefetch=prefetch
        self.max_chunkseq = self.config.get("chunkseq", 1)
        if self.prefetch > 0 and self.n_items > 0:
            self.try_prefetch(0)

    def load_config(self, kwargs: dict):
        with _FileLock(self.lock_file):
            config_file = resolve_file(self.url, f'{self.split_prefix}config.json', self.cache_dir)
        try:
            with open(config_file) as f:
                config = json.load(f)
        except BaseException as e:
            verbose_print(f'見つかりません {self.url} ({config_file})')
            config = dict(n_items=0, n_tokens=0)
        config.update(kwargs)
        return config
    
    def __len__(self):
        return self.n_items * self.n_subblocks

    def try_prefetch(self, chunkseq):
        chunk_file = chunkseq_to_filename(chunkseq % self.max_chunkseq, self.split_prefix, self.file_ext)
        resolve_file(self.url, chunk_file, self.cache_dir, sync=False)

    def get_chunks(self, chunk_file):
        if chunk_file in self.cache:
            return self.cache[chunk_file]
        with _FileLock(self.lock_file):
            chunk_file2 = resolve_file(self.url, chunk_file, self.cache_dir)
            chunks = load_chunk_file(chunk_file2)
        if chunks is None:
            # エラーで落ちるくらいなら、キャッシュのデータで学習を続ける
            chunks = self.cache[self.queue[0]]
        if self.shuffle:
            random.shuffle(chunks)
        if len(self.queue) == 64:
            older = self.queue.popleft()
            if older in self.cache:
                del self.cache[older]
        self.queue.append(chunk_file)
        self.cache[chunk_file] = chunks
        return chunks

    def __getitem__(self, index):
        offset = index % self.n_subblocks
        i = index // self.n_subblocks
        chunkseq = i // self.n_chunks
        chunk_file = chunkseq_to_filename(chunkseq, self.split_prefix, self.file_ext)
        chunks = self.get_chunks(chunk_file)
        if self.prefetch > 0 and index % self.n_chunks == 0:
            self.try_prefetch(chunkseq+self.prefetch)
        chunk = chunks[i % self.n_chunks]
        if self.n_subblocks > 1:
            return chunk[offset * self.block_size : (offset+1) * self.block_size]
        return chunk

    def compact(self, start, end):
        return start, end

    def reset(self):
        self.cache = {}
        for chunkpath in self.deque:
            os.remove(f'{self.cache_dir}/{chunkpath}')
        self.deque = deque(maxlen=64)

class FileDataset(Dataset):
    def __init__(self, tokenizer, filename, padding=False, max_length=DEFAULT_MAX_LENGTH):
        if isinstance(tokenizer, str):
            tokenizer = load_tokenizer(tokenizer)
        start = time.time()
        splitter = TextBlockSplitter(tokenizer=tokenizer, block_size=max_length)
        blocks = splitter.split_file(filename, N=-1)
        #splitter.report()
        # blocks = tokenize_file(tokenizer, filename=filename, N=-1, jsonl_key='text',
        #     block_size=max_length, padding=padding, overlap=0, sep=DEFAULT_SEP
        # )
        self.chunks = []
        token_counts = []
        for b in blocks:
                self.chunks.append(np.array(b, dtype=np.int32))
                token_counts.append(len(b))
        end = time.time()
        stat = stat_tokens(token_counts)
        self.n_tokens = stat['total']
        verbose_print(f'Loaded: {filename} Tokens: {self.n_tokens} {(end-start):.3f}s:', stat)

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, i):
        return self.chunks[i]

    def compact(self, start, end):
        length = end - start
        if length == len(self.chunks):
            return 0, length
        self.n_tokens  = self.n_tokens * length // len(self.chunks)
        self.chunks = self.chunks[start:end]
        return 0, length

def _parse_split(url, split):
    start, end = '', ''
    if '[' in url and url.endswith(']'):
        url, _, range = url.rpartition('[')
        start, end = range[:-1].split(':')
    if '?' in url:
        url, _, split = url.rpartition('?')
        if split.startswith('split='):
            split = split[6:]
    return url, split, start, end

def _parse_range(start, end, n_items):
    try:
        start = float(start)
        if start < 1:
            start = int(n_items * start)
        else:
            start = min(int(start), n_items)
        if start < 0:
            start = n_items - start
    except ValueError as e:
        start = 0
    try:
        end = float(end)
        if end < 1:
            end = int(n_items*end)
        else:
            end = min(int(end), n_items)
        if end < 0:
            end = n_items - end
    except ValueError:
        end = n_items
    if end - start > 0:
        return start, end
    return end, start

class Indexer(Dataset):
    def __init__(self, dataset, offset, length):
        self.dataset = dataset
        self.offset = offset
        self.length = length
        self.count = 0
        self.n_tokens = dataset.n_tokens

    def get_num_of_tokens(self):
        if len(self.dataset) == self.length:
            return self.n_tokens
        return self.n_tokens * self.length // len(self.dataset)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        idx = self.count
        self.count = (self.count + 1) % self.length
        return self.dataset[self.offset+idx]


def _make_mixer(datasets: List[Indexer]):
    lens = [len(ds) for ds in datasets]
    total = sum(lens)
    mixer_base = (total // min(lens))+1
    lens = [int((dlen * mixer_base) / total) for dlen in lens]
    verbose_print('Mixer:', lens)
    mixer = []
    for dlen, ds in zip(lens, datasets):
        mixer.extend([ds]*dlen)
    random.shuffle(mixer)
    return mixer


def build_inputs_for_clm(data, max_length):
    return torch.tensor(data[:max_length].astype(np.int64), dtype=torch.long)

def parse_url_list(url_list=[]):
    if isinstance(url_list, str):
        if os.path.exists(url_list):
            with open(url_list) as f:
                return [url.strip() for url in f.readlines() if url.strip() != '' and not url.startswith('#')]
        return url_list.split('|')
    return url_list

class DataComposer(Dataset):
    def __init__(self, url_list, format="pre", split = DEFAULT_SPLIT, 
                 max_length=DEFAULT_MAX_LENGTH, block_size=None,
                 build_fn=build_inputs_for_clm, tokenizer=None, shuffle=True,
                 cache_dir = DEFAULT_CACHE_DIR, use_filelock=True, prefetch=1):
        if block_size is None:
            self.max_length = max_length
            self.padding=True
        else:
            self.max_length = min(max_length, block_size)
            self.padding=False
        self.split = format + split
        self.cache_dir = f'{safe_dir(cache_dir)}/{random_name()}'
        os.makedirs(self.cache_dir, exist_ok=True)
        self.lock_file = f'{self.cache_dir}/lock' if use_filelock else None
        self.prefetch = prefetch
        self.build_fn = build_fn
        self._prepare_datasets(parse_url_list(url_list), tokenizer, block_size)

    def _prepare_datasets(self, urls, tokenizer, block_size):
        self.n_items = 0
        self.n_tokens = 0
        tokenizer_path = None
        datasets = []
        for i, url in enumerate(urls):
            url, split, start, end = _parse_split(url, self.split)
            if url.endswith('.gz') or url.endswith('.jsonl') or url.endswith('.txt'):
                if tokenizer is None:
                    verbose_print(f'トークンナイザーの指定がないので DEFAULT_TOKENIZER={DEFAULT_TOKENIZER}を使います')
                    tokenizer = load_tokenizer(DEFAULT_TOKENIZER)
                dataset = FileDataset(tokenizer, url, max_length=self.max_length, padding=self.padding)
            else:
                dataset = ChunkedDataset(url, split=split, block_size=block_size, 
                                         lock_file=self.lock_file, prefetch=self.prefetch,
                                         cache_dir=self.cache_dir)
                if tokenizer_path is None:
                    tokenizer_path = dataset.tokenizer_path
                else:
                    if tokenizer_path != dataset.tokenizer_path:
                        verbose_error(f'トークンナイザーが一致しません。{tokenizer_path}')
                        verbose_error(f'    {dataset.tokenizer_path} @{url}')
                        verbose_print(f'** {url} は、無視して学習を続けます。')
                        continue

            if len(dataset) == 0:
                verbose_print(f'** {url} は、無視して学習を続けます。')
                continue
            start, end = _parse_range(start, end, len(dataset))
            start, end = dataset.compact(start, end)
            ds = Indexer(dataset, start, end-start)
            self.n_items += len(ds)
            verbose_print(f'{url} トークン数: {ds.get_num_of_tokens():,}')
            self.n_tokens += ds.get_num_of_tokens()
            datasets.append(ds)
        verbose_print(f'Total tokens: {self.n_tokens:,}')
        self.mixer = _make_mixer(datasets)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.mixer = None
        if os.path.isdir(self.cache_dir):
            verbose_print(f'Cleaning up {self.cache_dir} ...')
            shutil.rmtree(self.cache_dir)

    def __len__(self):
        return self.n_items

    def __getitem__(self, idx):
        mix = len(self.mixer)
        item = self.mixer[idx % mix][idx]
        return self.build_fn(item, self.max_length)

## Seq2Seq

class MSP(object):
    def __init__(self, tokenizer, lambda_=3):
        self.lambda_ = lambda_
        self.eos_token_id = tokenizer.eos_token_id
        self.extra_ids = find_extra_ids(tokenizer)
        self.newline_id = find_newline_token_id(tokenizer)

    def __call__(self, data, max_length):
        data = data.tolist()
        inputs = tokens = []
        outputs = masked = []
        start=0
        index=0
        for length in np.random.poisson(self.lambda_, 1000):
            end = start+max(1, length)
            data_part = data[start:end]
            tokens.extend(data_part)
            if self.eos_token_id in data_part or self.newline_id in data_part:
                masked.extend(data_part)
            else:
                masked.append(self.extra_ids[index%100])
                index += 1
                tokens,masked = masked, tokens
            start = end
            if start > len(data):
                break
        return {
            "input_ids": torch.tensor(inputs[max_length//2], dtype=torch.long),
            "labels": torch.tensor(outputs[max_length//2], dtype=torch.long),
        }

class T5PretrainComposer(DataComposer):
    def __init__(self, url_list, split="train", **kwargs):
        DataComposer.__init__(self, url_list, format="pre", split=split, **kwargs)


class T5FinetuneComposer(DataComposer):
    def __init__(self, url_list, split="train", **kwargs):
        DataComposer.__init__(self, url_list, format="seq2seq", split=split, **kwargs)


class DP(object):
    def __init__(self, tokenizer, lambda_=20):
        self.lambda_ = lambda_
        self.eos_token_id = tokenizer.eos_token_id
        self.extra_ids = find_extra_ids(tokenizer)
        self.newline_id = find_newline_token_id(tokenizer)
        #print('@', self.newline_id, self.extra_ids)

    def __call__(self, data, max_length):
        index = 0
        start = 0
        size = min(max_length, len(data))
        for length in np.random.poisson(self.lambda_, 1000):
            start = start + max(1, length)
            if start >= size:
                break
            if data[start] != self.eos_token_id or data[start] != self.newline_id:
                data[start] = self.extra_ids[index]
                index+=1
        return torch.tensor(data[:max_length].astype(np.int64), dtype=torch.long)

class Seq2seq(object):
    def __init__(self, tokenizer):
        self.eos_token_id = tokenizer.eos_token_id

    def __call__(self, data, max_length):
        indices = np.where(data == self.eos_token_id)[0]
        assert indices.size > 1, "seq2seqは、</s>で区切られた２文からなるべきです。"
        index = indices[0]
        inputs = data[:index+1]
        labels = data[index+1:]
        if len(inputs)+len(labels) >= max_length:
            # 前半分と後ろ半分を連結する
            half_size = (max_length - len(labels)) // 2
            inputs = np.concatenate(inputs[:half_size], inputs[-half_size:])
        return {
            "input_ids": torch.tensor(inputs.astype(np.int64), dtype=torch.long),
            "labels": torch.tensor(labels.astype(np.int64), dtype=torch.long),
        }

"""
def build_inputs_for_seq2seq(data, max_length=None, target_max_length=None):
    target_max_length = target_max_length or max_length
    eos_id = data[-1]
    indices = np.where(data == SpecialTokenIds.SEP)[0]
    # index = data.tolist().index(SpecialTokenIds.SEP)
    if indices.size > 0:
        index = indices[0]
        inputs = data[:index+1]
        inputs[index] = eos_id
        labels = data[index+1:]
    else:
        inputs = []
        for x in data[:-1]:
            if random.random() < 0.3:
                if len(inputs) > 0 and inputs[-1] != SpecialTokenIds.MASK:
                    inputs.append(SpecialTokenIds.MASK)
                continue
            inputs.append(x)
        inputs.append(eos_id)
        inputs=np.array(inputs, dtype=np.int32)
        labels=data
    if isinstance(max_length, int) and len(inputs) > max_length:
        inputs = inputs[:max_length]
        inputs[max_length-1]=eos_id
    if isinstance(target_max_length, int) and len(labels) > target_max_length:
        labels = labels[:target_max_length]
        labels[target_max_length-1]=eos_id
    return {
        "input_ids": torch.tensor(inputs.astype(np.int64), dtype=torch.long),
        # "attention_mask": torch.tensor(inputs, dtype=torch.long),
        "labels": torch.tensor(labels.astype(np.int64), dtype=torch.long),
    }
"""