"""Read a user-owned external env file; never execute it or export its contents."""
import os
from pathlib import Path
import stat
import tempfile


KEY_FILE = Path.home() / '.config' / 'jev' / 'openrouter.env'


def load_openrouter_key(path=None):
    # Explicit per-process credentials keep precedence over the saved default.
    key = os.environ.get('OPENROUTER_API_KEY', '').strip()
    if key:
        return key
    path = Path(path) if path is not None else KEY_FILE
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    except FileNotFoundError:
        return None
    with os.fdopen(fd, 'r', encoding='utf-8') as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
            raise RuntimeError('Key file must be owned by this user with permissions 600.')
        lines = stream.read(16385)
    if len(lines) > 16384:
        raise RuntimeError('Key file is unexpectedly large.')
    result = None
    seen = False
    for line in lines.splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        if line.startswith('export '):
            line = line[7:].strip()
        name, sep, value = line.partition('=')
        if not sep or name.strip() != 'OPENROUTER_API_KEY' or seen:
            raise RuntimeError('Expected one OPENROUTER_API_KEY assignment in key file.')
        seen = True
        value = value.strip()
        if len(value) >= 2 and value[0] in ('"', "'") and value[-1] == value[0]:
            value = value[1:-1]
        if any(c.isspace() for c in value):
            raise RuntimeError('API key must be a single token.')
        result = value or None
    return result


def save_openrouter_key(key, path=None):
    key = key.strip()
    if not key or any(c.isspace() for c in key) or any(c in key for c in '\x00\'"'):
        raise ValueError('API key must be a nonempty single token without quotes.')
    path = Path(path) if path is not None else KEY_FILE
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if path.is_symlink():
        raise RuntimeError('Refusing to replace a symbolic link.')
    tmp = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent, delete=False) as stream:
            tmp = Path(stream.name)
            os.fchmod(stream.fileno(), 0o600)
            stream.write('OPENROUTER_API_KEY=' + key + '\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
    finally:
        if tmp is not None and tmp.exists():
            tmp.unlink()
