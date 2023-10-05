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

def getint_from_environ(key, given=None, default=None):
    if given:
        return int(given)
    try:
        return int(os.environ.get(key, default))
    except:
        return default


def format_unit(num: int, scale=1000)->str:
    """
    大きな数をSI単位系に変換して返す
    """
    if scale == 1024:
        if num < scale:
            return str(num)
        elif num < scale**2:
            return f"{num / scale:.1f}K"
        elif num < scale**3:
            return f"{num / scale**2:.1f}M"
        elif num < scale**4:
            return f"{num / scale**3:.1f}G"
        elif num < scale**5:
            return f"{num / scale**4:.1f}T"
        elif num < scale**6:
            return f"{num / scale**5:.1f}P"
        else:
            return f"{num / scale**6:.1f}Exa"
    elif scale == 60:
        if num < 1.0:
            return f"{num * 1000:.1f}ms"
        if num < scale:
            return f"{num:.1f}sec"
        elif num < scale**2:
            return f"{num / scale:.1f}min"
        elif num < (scale**2)*24:
            return f"{num / scale**2:.1f}hours"
        else:
            num2 = num % (scale**2)*24
            return f"{num // (scale**2)*24}days {num2 / scale**2:.1f}hours"
    else:
        if num < 1_000:
            return str(num)
        elif num < 1_000_000:
            return f"{num / 1_000:.1f}K"
        elif num < 1_000_000_000:
            return f"{num / 1_000_000:.1f}M"
        elif num < 1_000_000_000_000:
            return f"{num / 1_000_000_000:.1f}B"
        else:
            return f"{num / 1_000_000_000_000:.1f}T"

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
    print('🏙', *args, **kwargs)

def verbose_error(*args, **kwargs):
    """
    PaperTownのエラープリント
    """
    print('🌆', *args, **kwargs)
