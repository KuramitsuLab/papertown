from typing import List
import random 
import pandas as pd

from .papertown_tokenizer import find_ellipsis_token_id
from .papertown_utils import zopen, get_file_lines

DEFAULT_BLOCK_SIZE=1024
empty_tokens = []

class DefaultSplitter(object):

    def split(self, text:str, blocks: List[List[int]]):
        raise NotImplemented()

    def flush(self, blocks: List[List[int]]):
        pass

    def split_file(self, filename, update_fn=None, N=None): 
        if N == -1:
            N = get_file_lines(filename)
        if N:
            from tqdm import tqdm
            pbar = tqdm(total=N, desc=filename)
        blocks=[]
        with zopen(filename) as f:
            line = f.readline()
            c=1
            while line:
                self.split(line, blocks)
                if update_fn is not None:
                    update_fn(blocks)
                    blocks=[]
                line = f.readline()
                if N: 
                    pbar.update()
                    if c > N: break
                c+=1
        self.flush(blocks)
        if update_fn is not None:
            update_fn(blocks)
            blocks=[]
        if N:
            pbar.close()
        return blocks


class TextBlockSplitter(DefaultSplitter):
    def __init__(self, tokenizer, block_size, work_size=512, pad_size=64, sep=None):
        """
        pad_size: 空白埋めする最大のサイズ
        """
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id
        self.eos_token_id = tokenizer.eos_token_id
        self.ellipsis_token_id = find_ellipsis_token_id(tokenizer)
        self.block_size = block_size
        self.work_size = work_size
        self.work_buffers = []
        self.lazy_buffers = []
        self.pad_size = pad_size
        self.heads = [[] for _ in range(self.work_size//self.pad_size)]
        self.trancate_size=0
        self.sep = sep
        self.extra_tokens = empty_tokens
        self.pad_count = 0
        self.match_count = 0
        self.token_counts = []
        self.sep_counts = []

    def resize_as_dividable(self, tokens, size, trancate_size=0, padding_size=None):
        extra_size = len(tokens) % size 
        if extra_size == 0:
            return tokens
        if extra_size <= trancate_size:
            tokens = tokens[:-extra_size]
            if len(tokens)>2:
                tokens[-1] = self.eos_token_id
                if self.ellipsis_token_id:
                    tokens[-2] = self.ellipsis_token_id
            return tokens
        if padding_size is None or (size - extra_size) < padding_size:
            self.pad_count += (size - extra_size)
            return tokens + [self.pad_token_id] * (size - extra_size)
        return tokens[:(len(tokens)//size)*len(tokens)]

    def tokenize_nonsep(self, text:str):
        tokens = self.tokenizer.encode(text)
        self.token_counts.append(len(tokens))
        tokens = self.resize_as_dividable(tokens, self.pad_size, trancate_size=self.trancate_size)
        return tokens

    def tokenize_sep(self, text:str):
        text_blocks = text.split(self.sep)
        tokenizer = self.tokenizer
        chunks = [tokenizer.convert_tokens_to_ids(tokenizer.tokenize(line)) for line in text_blocks]
        chunks[-1] = tokenizer.build_inputs_with_special_tokens(chunks[-1])
        self.sep_counts.extend(len(c) for c in chunks)
        tokens = []
        chunk_bufs = []
        split_size = self.pad_size * 2
        for chunk in chunks:
            prev_len = len(chunk_bufs)
            chunk_bufs.extend(chunk)
            if len(chunk_bufs) < split_size:
                continue
            if len(chunk_bufs) % self.pad_size == 0:
                tokens.extend(chunk_bufs)
                chunk_bufs=[]
            else:
                splitted = self.resize_as_dividable(chunk_bufs, self.pad_size, 
                                                    padding_size=0, trancate_size=self.pad_size)
                tokens.extend(splitted)
                if prev_len > 0:
                    chunk_bufs = chunk
                else:
                    chunk_bufs = []
        return self.resize_as_dividable(tokens+chunk_bufs, self.pad_size, self.trancate_size)

    def add_buffer(self, blocks: List[List[int]], tokens: List[int]):
        assert(len(tokens) == self.work_size)
        if len(self.extra_tokens) > 0:
            self.lazy_buffers.append(tokens)
        elif len(self.lazy_buffers) > 0:
            lazy_buffers = self.lazy_buffers
            self.lazy_buffers = []
            for lazy_tokens in lazy_buffers:
                self.add_buffer(blocks, lazy_tokens)

        self.work_buffers.extend(tokens)
        if len(self.work_buffers) == self.block_size:
            blocks.append(self.work_buffers)
            self.work_buffers = []

    def push_head(self, tokens):
        index = len(tokens) // self.pad_size
        self.heads[index].append(tokens)

    def pop_head(self, extra_size):
        index = extra_size // self.pad_size
        if len(self.heads[index]) > 0:
            return self.heads[index].pop()
        return None

    def split(self, text:str, blocks: List[List[int]]):
        if self.sep is not None and self.sep in text:
            tokens = self.tokenize_sep(text)
        else:
            tokens = self.tokenize_nonsep(text)

        work_size = self.work_size
        if len(tokens) < work_size:
            head = self.pop_head(work_size - len(tokens) % work_size)
            if head:
                self.add_buffer(blocks, head+tokens)
            else:
                self.push_head(tokens)
            return
        tokens = self.extra_tokens + tokens
        for i in range(0, len(tokens) - work_size + 1, work_size):  
            segmented = tokens[i : i + work_size]
            self.add_buffer(blocks, segmented)

        extra_size = len(tokens) % work_size
        if extra_size == 0: # 最後の分割が揃っていればおしまい
            self.extra_tokens = empty_tokens
            return
        extra_tokens = tokens[-extra_size:]
        head = self.pop_head(work_size - extra_size)
        if head:
            self.add_buffer(blocks, extra_tokens+head)
            self.extra_tokens = empty_tokens
        else:
            self.extra_tokens = extra_tokens

    def flush(self, blocks: List[List[int]]):
        heads = []
        for hh in self.heads:
            heads.extend(hh)
        # print(pd.DataFrame({'heads': [len(h) for h in heads]}).describe())
        random.shuffle(heads)
        tokens = []
        for t in [self.extra_tokens]+heads:
            tokens.append(t)
        work_size = self.work_size
        for i in range(0, len(tokens) - work_size + 1, work_size):  
            segmented = tokens[i : i + work_size]
            self.add_buffer(blocks, segmented)

    def report(self):
        token_count = sum(self.token_counts)
        print(f'pad: {self.pad_count:,} {self.pad_count*100/token_count:.2f}%')
        print(f'match: {self.match_count:,} {len(self.heads)}')
        print(pd.DataFrame({'tokens': self.token_counts}).describe())
        # heads = [len(h) for hh in self.heads]
        # print(pd.DataFrame({'heads': heads}).describe())
        if len(self.sep_counts) > 0:
            print(pd.DataFrame({'separators': self.sep_counts}).describe())



"""
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
"""

