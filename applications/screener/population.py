"""Freeze native selections without collapsing repeated source occurrences."""

from contextlib import contextmanager
import csv
from dataclasses import dataclass
import hashlib
import gzip
from pathlib import Path
import re
import sqlite3
import tarfile
import tempfile

from applications.scoring.config import ConfigurationError
from applications.scoring.contracts import Entry, canonical, loads
from applications.scoring.runtime import checksum

MAX_MEMBER = 8 * 1024 * 1024


def read_control(path):
    config = {}
    for line in Path(path).read_text().splitlines():
        match = re.fullmatch(r'([a-zA-Z0-9_]+)\s*=\s*(.*?)\s*', line)
        if match:
            config[match[1]] = match[2]
    return config


def values(config, name):
    value = config[name]
    return value.split(':') if isinstance(value, str) else value


def source_config(source, scenario, *, completed):
    source = Path(source).resolve(strict=True)
    path = source / ('workflow/config.json' if completed else 'tools/templates/all.ctrl')
    config = loads(path.read_bytes()) if completed else read_control(path)
    names = values(config, 'docking_scenario_names')
    if names.count(scenario) != 1:
        raise ConfigurationError('SCENARIO', 'Choose exactly one existing named scenario')
    if int(values(config, 'docking_scenario_replicas')[names.index(scenario)]) != 1:
        raise ConfigurationError('REPLICAS', 'Screen integration requires one replica. Multiple replicas need an approved scientific aggregation rule.')
    if config['data_storage_mode'] != 'sharedfs' or config['job_storage_mode'] != 'sharedfs':
        raise ConfigurationError('STORAGE', 'Only same-host sharedfs native transport is qualified. S3 and cross-host coordination need separate adapters.')
    if config['ligand_library_format'] != 'smi':
        raise ConfigurationError('LIBRARY_FORMAT', 'Use native per-ligand .smi archives')
    if any(int(config.get(k, 0)) for k in ('run_atom_check', 'dynamic_tranche_filtering')):
        raise ConfigurationError('SELECTION', 'Disable native atom and tranche filters for per-ligand SMILES archives')
    return source, config, path


def collection_parts(key):
    parts = key.split('_')
    if len(parts) != 2 or any(not re.fullmatch(r'[A-Za-z0-9-]+', p) for p in parts):
        raise ConfigurationError('COLLECTION_ID', f'Unsupported native collection identity {key!r}')
    return parts


def archive_path(source, config, key):
    name, number = collection_parts(key)
    root = Path(config.get('sharedfs_collection_path', config['collection_folder']))
    if not root.is_absolute():
        root = source / 'tools' / root
    mode = config.get('object_store_data_collection_addressing_mode', config.get('data_collection_addressing_mode', 'metatranche'))
    if mode == 'hash':
        hashed = hashlib.sha256(f'{name}/{int(number):07}'.encode()).hexdigest()
        identifier = config.get('object_store_data_collection_identifier', config.get('data_collection_identifier', ''))
        return root / hashed[:2] / hashed[2:4] / identifier / 'smi' / name / f'{int(number):07}.tar.gz'
    return root / name / f'{number}.tar.gz'


@dataclass
class Population:
    path: Path
    database: sqlite3.Connection
    source: Path
    config: dict
    evidence: dict

    def verify(self):
        for path, sha in self.database.execute('SELECT path,sha FROM sources'):
            if checksum(path) != sha:
                raise ConfigurationError('SOURCE_CHANGED', f'Native source changed during preflight: {path}')

    def entries(self):
        for identifier, smiles, locator, metadata in self.database.execute('SELECT id,smiles,source,metadata FROM population ORDER BY ordinal'):
            yield Entry(identifier, smiles, locator, metadata)


@contextmanager
def snapshot(source, scenario, *, completed=False):
    source, config, config_path = source_config(source, scenario, completed=completed)
    selection = Path(config.get('collection_list_file', 'templates/todo.all'))
    if not selection.is_absolute():
        selection = source / 'tools' / selection
    # The original selection is required because grouped native JSON loses global CSV order.
    if not selection.is_file():
        raise ConfigurationError('SELECTION_MISSING', 'The original ordered collection_list_file is required; restore it before importing this screen')
    selection_sha, config_sha = checksum(selection), checksum(config_path)
    with tempfile.TemporaryDirectory(prefix='psichic-screen-population-') as temporary:
        root = Path(temporary)
        db = sqlite3.connect(root / 'population.sqlite3')
        try:
            db.executescript('''
                PRAGMA temp_store=FILE;
                CREATE TABLE selected(ordinal INTEGER PRIMARY KEY,collection TEXT,ligand TEXT,metadata TEXT);
                CREATE TABLE collections(collection TEXT PRIMARY KEY,count INTEGER);
                CREATE TABLE members(collection TEXT,ligand TEXT,smiles TEXT,original TEXT,sha TEXT,archive TEXT,
                                     member TEXT,PRIMARY KEY(collection,ligand));
                CREATE TABLE sparse(collection TEXT,ligand TEXT,original_collection TEXT,screen_index INTEGER,
                                    original TEXT,PRIMARY KEY(collection,ligand));
                CREATE TABLE population(ordinal INTEGER PRIMARY KEY,id TEXT,smiles TEXT,source TEXT,metadata TEXT);
                CREATE TABLE sources(path TEXT PRIMARY KEY,sha TEXT);
                CREATE TABLE summaries(collection TEXT,ligand TEXT,record TEXT);
                CREATE INDEX summary_identity ON summaries(collection,ligand);
            ''')
            db.executemany('INSERT INTO sources VALUES (?,?)', [(str(selection), selection_sha), (str(config_path), config_sha)])
            if completed:
                for summary in sorted((source / 'output-files' / scenario / 'csv').glob('*/*.csv.gz')):
                    summary_sha = checksum(summary)
                    with gzip.open(summary, 'rt', newline='') as handle:
                        reader = csv.DictReader(handle, strict=True)
                        for record in reader:
                            if None in record or 'collection_key' not in record or 'ligand' not in record:
                                raise ConfigurationError('NATIVE_SUMMARY', f'Malformed native summary {summary}')
                            db.execute('INSERT INTO summaries VALUES (?,?,?)',
                                       (record['collection_key'], record['ligand'], canonical({'source': str(summary), 'sha256': summary_sha, 'row': record})))
                    db.execute('INSERT INTO sources VALUES (?,?)', (str(summary), summary_sha))
            kind = config.get('collection_list_type', 'standard')
            prescreen = kind == 'standard' and int(config.get('prescreen_mode', 0)) == 1
            cutoff = int(config.get('prescreen_ligands_per_tranche', 0))
            if prescreen and cutoff < 1:
                raise ConfigurationError('SELECTION', 'Prescreen selection needs a positive prescreen_ligands_per_tranche')
            with selection.open(newline='', encoding='utf-8-sig') as handle:
                if kind == 'csv_collection_key_ligand':
                    reader = csv.DictReader(handle, strict=True)
                    names = reader.fieldnames or []
                    if len(names) != len(set(names)) or not {'collection_name', 'collection_number', 'ligand'} <= set(names):
                        raise ConfigurationError('SELECTION', 'Named selection needs unique headers and collection_name, collection_number, ligand')
                    for ordinal, row in enumerate(reader, 1):
                        if None in row or any(v is None for v in row.values()):
                            raise ConfigurationError('SELECTION', f'Malformed selection row {ordinal}')
                        key = row['collection_name'] + '_' + row['collection_number']
                        collection_parts(key)
                        db.execute('INSERT INTO selected VALUES (?,?,?,?)', (ordinal, key, row['ligand'], canonical(row)))
                        db.execute('INSERT OR IGNORE INTO collections VALUES (?,NULL)', (key,))
                elif kind == 'standard':
                    for ordinal, line in enumerate(handle, 1):
                        key, count = line.strip().split()
                        collection_parts(key)
                        if not count.isdecimal() or int(count) < 1:
                            raise ConfigurationError('SELECTION', 'Collection counts must be positive integers')
                        db.execute('INSERT INTO selected VALUES (?,?,NULL,?)', (ordinal, key, canonical({'native_line': line})))
                        db.execute('INSERT OR IGNORE INTO collections VALUES (?,?)', (key, int(count)))
                else:
                    raise ConfigurationError('SELECTION', f'Unsupported native selection type {kind!r}')
            archive_digest = hashlib.sha256()
            for key, count in db.execute('SELECT collection,count FROM collections ORDER BY rowid'):
                path = archive_path(source, config, key).resolve(strict=True)
                before = checksum(path)
                db.execute('INSERT OR IGNORE INTO sources VALUES (?,?)', (str(path), before))
                number = collection_parts(key)[1]
                members, listings = 0, 0
                with tarfile.open(path, 'r|gz') as archive:
                    for member in archive:
                        if member.isdir():
                            continue
                        parts = Path(member.name).parts
                        if (not member.isfile() or len(parts) != 2 or parts[0] != number or member.size > MAX_MEMBER
                                or not (parts[1].endswith('.smi') or parts[1] == '.listing')):
                            raise ConfigurationError('ARCHIVE', f'Unsafe or unsupported archive member {member.name!r}')
                        if parts[1] == '.listing':
                            listings += 1
                            if listings > 1:
                                raise ConfigurationError('ARCHIVE', 'Duplicate sparse .listing')
                            for line in archive.extractfile(member).read().decode('utf-8').splitlines():
                                original_collection, ligand, index = line.strip().split(',')
                                collection_parts(original_collection)
                                if not index.isdecimal():
                                    raise ConfigurationError('ARCHIVE', 'Sparse indices must be nonnegative integers')
                                try:
                                    db.execute('INSERT INTO sparse VALUES (?,?,?,?,?)', (key, ligand, original_collection, int(index), line))
                                except sqlite3.IntegrityError as error:
                                    raise ConfigurationError('ARCHIVE', f'Duplicate sparse ligand identity {ligand!r}') from error
                            continue
                        ligand = parts[1][:-4]
                        if not ligand or '.' in ligand or '\\' in ligand:
                            raise ConfigurationError('ARCHIVE', f'Ambiguous native ligand name {ligand!r}')
                        raw = archive.extractfile(member).read()
                        original = raw.decode('utf-8')
                        lines = original.splitlines()
                        if len(lines) > 1:
                            raise ConfigurationError('ARCHIVE', f'Expected one SMILES line in {member.name}')
                        smiles = original.split()[0] if original.split() else ''
                        try:
                            db.execute('INSERT INTO members VALUES (?,?,?,?,?,?,?)',
                                       (key, ligand, smiles, original, before, str(path), member.name))
                        except sqlite3.IntegrityError as error:
                            raise ConfigurationError('ARCHIVE', f'Duplicate ligand identity in {path}: {ligand}') from error
                        members += 1
                if not members or not prescreen and count is not None and members != count:
                    raise ConfigurationError('ARCHIVE_COVERAGE', f'Collection {key} expected {count} entries but archive contains {members}')
                if prescreen:
                    if listings != 1:
                        raise ConfigurationError('ARCHIVE', 'Sparse archive requires one .listing')
                    missing = db.execute('''SELECT s.ligand FROM sparse s LEFT JOIN members m
                        ON m.collection=s.collection AND m.ligand=s.ligand WHERE s.collection=? AND m.ligand IS NULL LIMIT 1''', (key,)).fetchone()
                    if missing:
                        raise ConfigurationError('MISSING_IDENTITY', f'Sparse listing identity {key}/{missing[0]} is absent from its archive')
                if checksum(path) != before:
                    raise ConfigurationError('SOURCE_CHANGED', f'Archive changed during import: {path}')
                archive_digest.update((canonical({'collection': key, 'path': str(path), 'sha256': before}) + '\n').encode())
            ordinal = 0
            for occurrence, collection, ligand, selection_metadata in db.execute('SELECT * FROM selected ORDER BY ordinal'):
                if prescreen:
                    rows = db.execute('''SELECT m.ligand,m.smiles,m.original,m.sha,m.archive,m.member,s.original_collection,s.screen_index,s.original
                        FROM sparse s JOIN members m ON m.collection=s.collection AND m.ligand=s.ligand
                        WHERE s.collection=? AND s.screen_index<? ORDER BY s.rowid''', (collection, cutoff))
                else:
                    rows = db.execute('''SELECT ligand,smiles,original,sha,archive,member,collection,NULL,NULL FROM members
                        WHERE collection=? AND (? IS NULL OR ligand=?) ORDER BY rowid''', (collection, ligand, ligand))
                found = False
                for name, smiles, original, sha, archive, member, original_collection, index, listing in rows:
                    found = True
                    ordinal += 1
                    metadata = {'collection': original_collection, 'ligand': name, 'scenario': scenario, 'replica': 0,
                                'source_occurrence': occurrence, 'original_text': original,
                                'selection': loads(selection_metadata), 'archive_sha256': sha, 'archive': archive, 'member': member,
                                'source_archive_collection': collection, 'prescreen_index': index, 'prescreen_listing': listing}
                    metadata['native_summaries'] = [loads(value[0]) for value in db.execute(
                        'SELECT record FROM summaries WHERE collection=? AND ligand=?', (original_collection, name))]
                    locator = f'{selection}#occurrence={occurrence}&member={member}'
                    db.execute('INSERT INTO population VALUES (?,?,?,?,?)', (ordinal, name, smiles, locator, canonical(metadata)))
                if not found and not prescreen:
                    raise ConfigurationError('MISSING_IDENTITY', f'Selected identity {collection}/{ligand} is absent from its archive')
            if not ordinal:
                raise ConfigurationError('EMPTY_POPULATION', 'Native selection contains no entries')
            if checksum(selection) != selection_sha or checksum(config_path) != config_sha:
                raise ConfigurationError('SOURCE_CHANGED', 'Native source configuration or selection changed during import')
            path = root / 'candidates.csv'
            with path.open('w', newline='') as handle:
                writer = csv.writer(handle)
                writer.writerow(['ID', 'SMILES'])
                writer.writerows(db.execute('SELECT id,smiles FROM population ORDER BY ordinal'))
            population = Population(path, db, source, config, {
                'schema_version': 1, 'source_run': str(source), 'scenario': scenario, 'replica': 0,
                'rows': ordinal, 'selection': str(selection), 'selection_sha256': selection_sha,
                'config': str(config_path), 'config_sha256': config_sha, 'archive_inventory_sha256': archive_digest.hexdigest(),
                'completion': 'caller-declared' if completed else 'new-screen', 'legacy_scores_reused': False})
            yield population
            population.verify()
        finally:
            db.close()
