from typing import List
import random
import json
import numpy as np
import pandas as pd

from .papertown_tokenizer import find_ellipsis_token_id, find_newline_token_id
from .papertown_utils import zopen, get_file_lines, verbose_print

DEFAULT_BLOCK_SIZE=1024
DEFAULT_SEQ2SEQ_SEP='<outpuT>'

empty_tokens = []

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

def parse_strip(s):
    return s.strip().replace('<nL>', '\n')

def parse_jsonl(line):
    d = json.loads(line)
    if 'out' in d:
        return f"{d['in']}{DEFAULT_SEQ2SEQ_SEP}{d['out']}"
    return d['text']

def file_iterator(filename, N=None):
    if N == -1:
        N = get_file_lines(filename)-1
    if N:
        from tqdm import tqdm
        pbar = tqdm(total=N, desc=filename)
    parse_fn = parse_strip
    if '.json' in filename:
        parse_fn = parse_jsonl
    c=0
    with zopen(filename) as f:
        line = f.readline()
        while line is not None:
            c+=1
            if N: 
                pbar.update()
                if c > N: break
            yield parse_fn(line)
            line = f.readline()
    if N:
        pbar.close()


class DefaultSplitter(object):
    def __init__(self, tokenizer, block_size, sep=None):
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id
        self.eos_token_id = tokenizer.eos_token_id
        self.ellipsis_token_id = find_ellipsis_token_id(tokenizer)
        self.block_size = block_size
        self.sep = sep if sep != '' else None
        self.token_counts = []
        self.pad_count = 0

    def split(self, text:str, blocks: List[List[int]]):
        raise NotImplemented()

    def flush(self, blocks: List[List[int]]):
        pass

    def split_iter(self, iterator, update_fn=None):
        blocks = []
        if update_fn:
            for text in iterator:
                self.split(text, blocks)
                update_fn(blocks)
                blocks=[]
            self.flush(blocks)
            update_fn(blocks)
        else:
            for text in iterator:
                self.split(text, blocks)
            self.flush(blocks)
            return blocks

    def report(self, logs: dict = None, verbose=True):
        token_count = sum(self.token_counts)
        if logs:
            logs['n_tokens'] = token_count
            logs['tokens'] = stat_tokens(self.token_counts)
        if verbose:
            print(pd.DataFrame({'tokens': self.token_counts}).describe())
        if self.pad_count > 0:
            if verbose:
                print(f'pad: {self.pad_count:,} {self.pad_count*100/token_count:.2f}%')
            if logs:
                logs['padding_rate'] = self.pad_count / token_count


class SimpleTextBlockSplitter(DefaultSplitter):
    def __init__(self, tokenizer, block_size, sep=None):
        super().__init__(tokenizer, block_size, sep=sep)
        self.extra_tokens=empty_tokens
        self.split_prefix='pre'

    def tokenize_text(self, text:str, blocks: List[List[int]]):
        tokens = self.tokenizer.encode(text)
        work_size = self.block_size
        tokens = self.extra_tokens + tokens
        for i in range(0, len(tokens) - work_size + 1, work_size):  
            segmented = tokens[i : i + work_size]
            blocks.append(segmented)
            self.token_counts.append(len(segmented))
        extra_size = len(tokens) % work_size
        if extra_size == 0:
            self.extra_tokens = empty_tokens
        else:
            self.extra_tokens = tokens[-extra_size:]

    def split(self, text:str, blocks: List[List[int]]):
        self.tokenize_text(text, blocks)


class MultiTextBlockSplitter(DefaultSplitter):
    def __init__(self, tokenizer, block_size, sep=None, work_size=512, pad_size=64):
        super().__init__(tokenizer, block_size, sep=sep)
        self.split_prefix='pre'
        self.work_size = work_size
        self.trancate_size=0
        self.pad_size = pad_size
        # 警告が出る場合があるので、padding を変更する
        self.pad_token_id = find_newline_token_id(tokenizer)
        self.extra_tokens = empty_tokens
        self.work_buffers = []
        self.lazy_buffers = []
        self.heads = [[] for _ in range(self.work_size//self.pad_size)]
        self.match_count = 0
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
        counts = [len(c) for c in chunks]
        self.token_counts.append(sum(counts))
        self.sep_counts.extend(counts)
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
                splitted = self.resize_as_dividable(chunk_bufs, self.pad_size, trancate_size=self.pad_size, padding_size=0)
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

class SimpleTextSplitter(DefaultSplitter):
    def __init__(self, tokenizer, block_size, sep=None):
        super().__init__(tokenizer, block_size, sep=sep)
        self.split_prefix=''

    # def tokenize_pair(self, text:str, text_pair: str, blocks: List[List[int]]):
    #     inputs = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(text))
    #     labels = self.tokenizer.encode(text_pair)
    #     if len(labels) > self.block_size:
    #         # ラベルの方が大きい場合は諦める
    #         return
    #     if len(inputs)+len(labels) > self.block_size:
    #         half_size = (self.block_size - len(labels)) // 2
    #         prefix = inputs[:half_size]
    #         suffix = inputs[-half_size:]
    #         if self.ellipsis_token_id:
    #             prefix[-1] = self.ellipsis_token_id
    #         inputs = prefix + suffix
    #     blocks.append(inputs+labels)
    #     self.token_counts.append(len(inputs+labels))

    def split(self, text:str, blocks: List[List[int]]):
        inputs = self.tokenizer.encode(text)
        if len(inputs) > self.block_size:
            half_size = self.block_size // 2
            prefix = inputs[:half_size]
            suffix = inputs[-half_size:]
            if self.ellipsis_token_id:
                prefix[-1] = self.ellipsis_token_id
            inputs = prefix + suffix
        blocks.append(inputs)
        inputs_size = len(inputs)
        self.token_counts.append(inputs_size)
        self.pad_count += self.block_size - inputs_size


class TextPairSplitter(DefaultSplitter):
    def __init__(self, tokenizer, block_size, sep=None):
        super().__init__(tokenizer, block_size, sep=sep)
        self.split_prefix='seq2seq'

    def tokenize_pair(self, text:str, text_pair: str, blocks: List[List[int]]):
        inputs = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(text))
        labels = self.tokenizer.encode(text_pair)
        if len(labels) > self.block_size:
            # ラベルの方が大きい場合は諦める
            return
        if len(inputs)+len(labels) > self.block_size:
            trimmed_size = self.block_size - len(labels)
            inputs = inputs[:trimmed_size]
            inputs[-1] = self.eos_token_id
        blocks.append(inputs+labels)
        self.token_counts.append(len(inputs+labels))

    def tokenize_pair(self, text:str, text_pair: str, blocks: List[List[int]]):
        inputs = self.tokenizer.encode(text)
        labels = self.tokenizer.encode(text_pair)
        if len(labels) > self.block_size:
            # ラベルの方が大きい場合は諦める
            return
        if len(inputs)+len(labels) > self.block_size:
            # ラベルは完全に残す
            half_size = (self.block_size - len(labels)) // 2
            prefix = inputs[:half_size]
            suffix = inputs[-half_size:]
            if self.ellipsis_token_id:
                prefix[-1] = self.ellipsis_token_id
            inputs = prefix + suffix
        blocks.append(inputs+labels)
        self.token_counts.append(len(inputs+labels))

    def split(self, text:str, blocks: List[List[int]]):
        t = text.split(self.sep)
        if len(t)==2:
            self.tokenize_pair(t[0], t[1], blocks)
        else:
            raise ValueError(f'In text {repr(text)}, the {self.sep} token is required.')


def new_TextSplitter(tokenizer, format='simple', block_size=1024, padding=True, sep=None):
    if format == 'simple':
        if padding:
            splitter = SimpleTextSplitter(tokenizer, block_size=block_size, sep=sep)
        else:
            splitter = SimpleTextBlockSplitter(tokenizer, block_size=block_size, sep=sep)
    elif format == 'multi':
        splitter = MultiTextBlockSplitter(tokenizer, block_size=block_size, sep=sep)
    elif format == 'seq2seq':
        if not padding:
            verbose_print("format='seq2seq'では、padding=Falseは無効です。")
        splitter = TextPairSplitter(tokenizer, block_size=block_size, sep=DEFAULT_SEQ2SEQ_SEP)
    return splitter
