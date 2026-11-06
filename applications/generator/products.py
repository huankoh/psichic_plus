"""Freeze native product rows without inferring search or prediction provenance."""

from contextlib import contextmanager
from dataclasses import dataclass
import csv
import hashlib
import os
from pathlib import Path
import stat
import tempfile

from applications.scoring.config import ConfigurationError
from applications.scoring.contracts import Entry, canonical
from applications.scoring.runtime import checksum


def file_identity(value):
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)


@dataclass(frozen=True)
class ProductSnapshot:
    source: Path
    path: Path
    sha256: str
    identity: tuple
    columns: tuple[str, ...]
    row_count: int

    def verify(self):
        if (file_identity(self.source.stat(follow_symlinks=False)) != self.identity
                or checksum(self.source) != self.sha256
                or file_identity(self.source.stat(follow_symlinks=False)) != self.identity):
            raise ConfigurationError('SOURCE_CHANGED', 'Native molecules.csv changed during import. No final inference was started.',
                                     'Stop the native writer and import its completed products into a new run.')
        if checksum(self.path) != self.sha256:
            raise ConfigurationError('SOURCE_CHANGED', 'The private product snapshot changed during import')

    def entries(self, id_column, smiles_column, generation):
        with self.path.open(newline='', encoding='utf-8-sig') as handle:
            for index, row in enumerate(csv.DictReader(handle, strict=True), 1):
                yield Entry(row[id_column], row[smiles_column], f'{self.source}#row={index}',
                            canonical({'native': row, 'generation': generation}))


@contextmanager
def snapshot(source_run, id_column='node_id', smiles_column='smiles'):
    source = Path(source_run).expanduser().resolve() / 'molecules.csv'
    with tempfile.TemporaryDirectory(prefix='psichic-products-') as directory:
        destination = Path(directory) / 'molecules.csv'
        fd = os.open(source, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(fd, 'rb') as original, destination.open('xb') as retained:
            before = os.fstat(original.fileno())
            if not stat.S_ISREG(before.st_mode):
                raise ConfigurationError('INVALID_SOURCE', 'Native molecules.csv must be a regular file')
            content = hashlib.sha256()
            while block := original.read(1024 * 1024):
                content.update(block)
                retained.write(block)
            retained.flush()
            os.fsync(retained.fileno())
            if file_identity(before) != file_identity(os.fstat(original.fileno())):
                raise ConfigurationError('SOURCE_CHANGED', 'Native molecules.csv changed while being copied')
        with destination.open(newline='', encoding='utf-8-sig') as handle:
            reader = csv.DictReader(handle, strict=True)
            columns = reader.fieldnames or []
            if not columns or any(not name for name in columns) or len(set(columns)) != len(columns):
                raise ConfigurationError('INVALID_PRODUCTS', 'Native products need unique, nonempty CSV headers')
            required = {id_column, smiles_column, 'score', 'num_reactions'}
            if id_column == smiles_column or not required <= set(columns):
                raise ConfigurationError('INVALID_PRODUCTS',
                                         f'Native products require distinct ID and SMILES columns and columns {sorted(required)}')
            count = 0
            for count, row in enumerate(reader, 1):
                if None in row or any(value is None for value in row.values()):
                    raise ConfigurationError('INVALID_PRODUCTS', f'Malformed native product row {count}')
            if not count:
                raise ConfigurationError('EMPTY_POPULATION', 'Native molecules.csv contains no completed product rows')
        value = ProductSnapshot(source, destination, content.hexdigest(), file_identity(before), tuple(columns), count)
        value.verify()
        yield value
