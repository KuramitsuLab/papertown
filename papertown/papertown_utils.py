import os
import gzip

def safe_dir(dir):
    if dir.endswith('/'):
        dir = dir[:-1]
    return dir

def safe_join(dir, file):
    if dir.endswith('/'):
        dir = dir[:-1]
    if file.startswith('/'):
        file = file[1:]
    return f'{dir}/{file}'

DEFAULT_TOKENIZER = os.environ.get('PT_TOKENIZER', 'kkuramitsu/kawagoe')
DEFAULT_SPLIT='train'
DEFAULT_CACHE_DIR = safe_dir(os.environ.get('PT_CACHE_DIR', '.'))


def zopen(filepath):
    if filepath.endswith('.gz'):
        return gzip.open(filepath, 'rt')
    else:
        return open(filepath, 'r')

def get_file_lines(filepath):
    with zopen(filepath) as f:
        line = f.readline()
        c=1
        while line:
            line = f.readline()
            c+=1
    return c


def verbose_print(*args, **kwargs):
    """
    PaperTown 用のデバッグプリント
    """
    print('📃', *args, **kwargs)

def verbose_error(*args, **kwargs):
    """
    PaperTownのエラープリント
    """
    print('💔', *args, **kwargs)
