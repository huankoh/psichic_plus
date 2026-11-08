"""Frozen inputs and scoped rebuild evidence for Triton 3.3.1 host extensions."""

from dataclasses import dataclass, field
import importlib.util
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import sysconfig

if __package__:
    from .runtime import RuntimeMismatch, checksum, file_evidence, stamp, verify_files
else:
    from runtime import RuntimeMismatch, checksum, file_evidence, stamp, verify_files

CONTRACT = 'triton-3.3.1-host-rebuild-v1'
CONTROLS = ('CC', 'PATH', 'LD_LIBRARY_PATH', 'LIBRARY_PATH', 'CPATH', 'C_INCLUDE_PATH',
            'GCC_EXEC_PREFIX', 'COMPILER_PATH', 'SOURCE_DATE_EPOCH', 'LD_PRELOAD',
            'TRITON_LIBCUDA_PATH', 'TRITON_CUDACRT_PATH', 'TRITON_CUDART_PATH')
FORBIDDEN = ('TRITON_CACHE_MANAGER', 'TRITON_REMOTE_CACHE_BACKEND', 'TRITON_KERNEL_OVERRIDE',
             'TRITON_OVERRIDE_DIR', 'TRITON_KERNEL_DUMP', 'TRITON_DUMP_DIR', 'LD_PRELOAD',
             'GCC_EXEC_PREFIX', 'COMPILER_PATH', 'CPATH', 'C_INCLUDE_PATH', 'LIBRARY_PATH',
             'TRITON_CUDACRT_PATH', 'TRITON_CUDART_PATH')


def command(args, *, input=None):
    return subprocess.run(args, input=input, capture_output=True, text=True, check=True, timeout=60)


def controls():
    for name in FORBIDDEN:
        if os.environ.get(name):
            raise RuntimeMismatch(f'Unqualified native build control: {name}')
    names = set(CONTROLS) | {name for name in os.environ if name.startswith('TRITON_') and name != 'TRITON_CACHE_DIR'}
    return {name: os.environ.get(name) for name in sorted(names)}


def same_code(actual, expected):
    # Triton's registry loads driver.py again under the module name "nvidia".
    return actual.co_filename == expected.co_filename and actual == expected


def policy_identity(policy):
    if policy is None:
        return None
    return {**policy, 'files': [{'name': f['name'], 'sha256': f['sha256']} for f in policy['files']]}


def verify_inputs(policy):
    for item in policy['files']:
        try:
            if stamp(item['path']) != item['stamp'] and checksum(item['path']) != item['sha256']:
                raise RuntimeMismatch(f"Native input changed or disappeared: {item['path']}")
        except OSError as error:
            raise RuntimeMismatch(f"Native input changed or disappeared: {item['path']}") from error


def freeze():
    """Inspect inputs without compiling, loading CUDA, or creating a cache."""
    if importlib.util.find_spec('triton') is None:
        return None
    import triton
    from triton.backends.nvidia import driver

    if triton.__version__ != '3.3.1' or sys.platform != 'linux':
        raise RuntimeMismatch('Native generation requires qualified Linux Triton 3.3.1')
    env = controls()
    cc = shutil.which(os.environ.get('CC') or 'gcc')
    if not cc:
        raise RuntimeMismatch('Native generation requires a GCC executable')
    version = command([cc, '--version']).stdout
    if 'gcc' not in version.lower():
        raise RuntimeMismatch('Native generation requires GCC')
    cc = str(Path(cc).resolve())
    include_query = command([cc, '-E', '-v', '-x', 'c', '-'], input='').stderr
    include_text = include_query.split('#include <...> search starts here:\n')[1].split('End of search list.')[0]
    include = [str(Path(line.strip()).resolve()) for line in include_text.splitlines() if line.strip()]
    scheme = sysconfig.get_default_scheme()
    scheme = 'posix_prefix' if scheme == 'posix_local' else scheme
    python_include = sysconfig.get_paths(scheme=scheme)['include']
    include += [python_include, *driver.include_dir]
    programs = {name: str(Path(shutil.which(value) or value).resolve(strict=True))
                for name in ('cc1', 'collect2', 'as', 'ld', 'lto-wrapper')
                for value in [command([cc, '-print-prog-name=' + name]).stdout.strip()]}
    compiler_lib = Path(command([cc, '-print-libgcc-file-name']).stdout.strip()).resolve().parent
    sysroot = command([cc, '-print-sysroot']).stdout.strip()
    paths = {Path(cc), Path(sysconfig.__file__).resolve(),
             Path(importlib.util.find_spec(sysconfig._get_sysconfigdata_name()).origin).resolve(),
             *(Path(p) for p in programs.values())}
    for root in [Path(triton.__file__).parent, compiler_lib, Path(programs['cc1']).parent,
                 *(Path(p) for p in include), *([Path(sysroot)] if sysroot else [])]:
        paths.update(p.resolve() for p in root.rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    library_dirs = list(driver.library_dirs())
    for name in ('crti.o', 'crtn.o', 'crtbeginS.o', 'crtendS.o', 'libgcc.a', 'libgcc_s.so',
                 'libgcc_s.so.1', 'libc.so', 'libc.so.6', 'libc_nonshared.a', 'libm.so', 'libm.so.6',
                 'libdl.so', 'libdl.so.2', 'libpthread.so', 'libpthread.so.0'):
        path = Path(command([cc, '-print-file-name=' + name]).stdout.strip())
        if path.is_file():
            paths.add(path.resolve())
    for directory in library_dirs:
        paths.update(p.resolve() for p in Path(directory).glob('libcuda.so*') if p.is_file())
    if not any(p.name.startswith('libcuda.so') for p in paths):
        raise RuntimeMismatch('No CUDA driver link input found without initializing CUDA')
    # The compiler processes and linked driver depend on host shared libraries too.
    for path in tuple(paths):
        if path in {Path(cc), *(Path(p) for p in programs.values())} or path.name.startswith('libcuda.so'):
            result = subprocess.run(['ldd', str(path)], capture_output=True, text=True, timeout=30)
            for value in re.findall(r'(/[^\s()]+)', result.stdout):
                if Path(value).is_file():
                    paths.add(Path(value).resolve())
    for path in (Path('/etc/ld.so.cache'), Path('/etc/ld.so.conf'), Path('/usr/bin/ldd')):
        if path.is_file():
            paths.add(path.resolve())
    paths.update(p.resolve() for p in Path('/etc/ld.so.conf.d').glob('*') if p.is_file())
    files = [file_evidence('native-input/' + str(p), p, allow_empty=True) for p in sorted(paths)]
    return {'contract': CONTRACT, 'version': version, 'compiler': cc, 'programs': programs,
            'environment': env, 'include_dirs': include, 'library_dirs': library_dirs,
            'backend_include_dirs': list(driver.include_dir), 'python_include_dir': python_include,
            'extension_suffix': sysconfig.get_config_var('EXT_SUFFIX'),
            'platform_key': driver.platform_key(), 'files': files}


@dataclass
class Build:
    module: str
    source: str
    generator: dict
    directory: Path
    output: str | None = None
    sha256: str | None = None
    artifacts: dict = field(default_factory=dict)
    command: list = field(default_factory=list)
    dependencies: list = field(default_factory=list)
    loaded: bool = False


class NativeBuilds:
    """Observe upstream build boundaries and authorize only independently rebuilt bytes.

    This is a file-provenance check in a dedicated worker, not memory attestation.
    The profile callback is restored on exit and no upstream function is replaced.
    """

    def __init__(self, profile, directory):
        from triton.backends.nvidia import driver
        from triton.runtime import build, cache

        self.profile = profile
        self.policy = profile['native_generation']
        self.directory = Path(directory).resolve()
        self.driver, self.builder, self.cache = driver, build, cache
        self.pending = {}
        self.launchers = {}
        self.records = []
        self.paths = {}
        self.known = {f['path']: f for f in self.policy['files'] + profile['source']}
        self.previous = None
        self.failed = False
        self.complete = False

    def check_inputs(self):
        if self.policy['contract'] != CONTRACT or controls() != self.policy['environment']:
            raise RuntimeMismatch('Native generation policy or controls changed')
        compiler = shutil.which(os.environ.get('CC') or 'gcc')
        if compiler is None or str(Path(compiler).resolve()) != self.policy['compiler']:
            raise RuntimeMismatch('Native compiler selection changed')
        verify_files(self.profile['source'])
        verify_inputs(self.policy)

    def __enter__(self):
        self.check_inputs()
        if sys.getprofile() is not None:
            raise RuntimeMismatch('Native build observation requires an unoccupied worker profiler')
        cache_path = Path(os.environ.get('TRITON_CACHE_DIR', '')).resolve()
        if cache_path != self.directory / 'cache':
            raise RuntimeMismatch('Native cache must belong to this invocation')
        self.directory.mkdir()
        cache_path.mkdir()
        sys.setprofile(self.observe)
        return self

    def __exit__(self, kind, value, tb):
        active = sys.getprofile()
        sys.setprofile(self.previous)
        if active != self.observe or self.failed or self.pending or self.launchers:
            raise RuntimeMismatch('Native build observation did not complete')
        self.complete = True

    def _known_bytes(self, path):
        path = str(Path(path).resolve())
        item = self.known.get(path)
        if item is None or checksum(path) != item['sha256']:
            raise RuntimeMismatch(f'Unrecorded or changed native input: {path}')
        return {'path': path, 'sha256': item['sha256']}

    def _launcher(self, frame):
        source, meta = frame.f_locals['src'], frame.f_locals['metadata']
        fn = getattr(getattr(source, 'fn', None), 'fn', None)
        if fn is None or not hasattr(fn, '__code__'):
            raise RuntimeMismatch('Launcher must originate from a frozen JIT Python function')
        origin = self._known_bytes(fn.__code__.co_filename)
        signature = source.signature
        keys = list(signature)
        if keys not in (list(source.fn.arg_names), list(range(len(source.fn.arg_names)))):
            raise RuntimeMismatch('Launcher signature does not match the frozen kernel arguments')
        def signature_type(value):
            if isinstance(value, tuple):
                return [signature_type(item) for item in value]
            if not isinstance(value, str) or not re.fullmatch(r'\*?k?[A-Za-z][A-Za-z0-9_]*', value):
                raise RuntimeMismatch('Unsupported native launcher signature')
            if not value.startswith('*') and value != 'constexpr':
                self.driver.ty_to_cpp(value)
            return value
        signature = [[list(key) if isinstance(key, tuple) else key, signature_type(value)]
                     for key, value in signature.items()]
        options = meta._asdict() if hasattr(meta, '_asdict') else vars(meta)
        # Kernel metadata is runtime evidence, never a late extension to profile identity.
        options = json.loads(json.dumps(options, default=lambda v: vars(v), allow_nan=False))
        self.launchers[id(frame)] = {'kind': 'launcher', 'kernel': fn.__module__ + '.' + fn.__qualname__,
                                    'kernel_source': origin, 'signature': signature, 'options': options}

    def _request(self, frame):
        self.check_inputs()
        source, name = frame.f_locals['src'], frame.f_locals['name']
        caller = frame.f_back
        if name == 'cuda_utils' and same_code(caller.f_code, self.driver.CudaUtils.__init__.__code__):
            expected = Path(self.driver.dirname, 'driver.c').read_text()
            generator = {'kind': 'utility', 'source': self._known_bytes(Path(self.driver.dirname, 'driver.c'))}
        elif name == '__triton_launcher' and same_code(caller.f_code, self.driver.CudaLauncher.__init__.__code__):
            generator = self.launchers.get(id(caller))
            if generator is None:
                raise RuntimeMismatch('Unobserved native launcher signature')
            expected = self.driver.make_launcher(caller.f_locals['constants'], caller.f_locals['signature'])
        else:
            raise RuntimeMismatch(f'Unauthorized native generator: {name}')
        if source != expected:
            raise RuntimeMismatch('Native source differs from the authorized generator')
        folder = self.directory / f'build-{len(self.records):04d}'
        folder.mkdir()
        (folder / 'input.c').write_text(source)
        record = Build(name, source, generator, folder)
        self.pending[id(frame)] = record
        self.records.append(record)

    def _build_start(self, frame, record):
        args = frame.f_locals
        if (args['name'] != record.module or Path(args['src']).read_text() != record.source
                or args['library_dirs'] != self.policy['library_dirs']
                or args['include_dirs'] != self.policy['backend_include_dirs'] or args['libraries'] != ['cuda']):
            raise RuntimeMismatch('Native compiler inputs or options changed')
        include = [*args['include_dirs'], args['srcdir'], self.policy['python_include_dir']]
        record.command = [self.policy['compiler'], args['src'], '-O3', '-shared', '-fPIC', '-Wno-psabi',
                          '-o', str(Path(args['srcdir']) / (record.module + self.policy['extension_suffix'])),
                          '-lcuda', *[f'-L{p}' for p in args['library_dirs']], *[f'-I{p}' for p in include]]
        query = [self.policy['compiler'], args['src'], '-M', *[f'-I{p}' for p in include]]
        deps = command(query).stdout
        (record.directory / 'dependencies.txt').write_text(deps)
        paths = shlex.split(deps.replace('\\\n', ' ').split(':', 1)[1])
        record.dependencies = [self._known_bytes(p) for p in paths if Path(p).resolve() != Path(args['src']).resolve()]
        record.artifacts['source_path'] = args['src']
        record.artifacts['build_arguments'] = {key: args[key] for key in
                                             ('name', 'src', 'srcdir', 'library_dirs', 'include_dirs', 'libraries')}

    def _build_end(self, frame, output, record):
        self.check_inputs()
        args = record.artifacts['build_arguments']
        if not output or Path(args['src']).read_text() != record.source:
            raise RuntimeMismatch('Native compilation failed or source changed')
        observed = frame.f_locals['cc_cmd']
        resolved = shutil.which(observed[0])
        if resolved is None or [str(Path(resolved).resolve()), *observed[1:]] != record.command:
            raise RuntimeMismatch('Observed native compiler command differs from the frozen options')
        record.artifacts['observed_command'] = observed
        verify = record.directory / 'rebuild'
        verify.mkdir()
        # Keep the original source path: driver.c embeds __FILE__ in its diagnostics.
        rebuilt = self.builder._build(args['name'], args['src'], str(verify), args['library_dirs'],
                                      args['include_dirs'], args['libraries'])
        actual_hash = checksum(output)
        if checksum(rebuilt) != actual_hash:
            raise RuntimeMismatch('Native binary differs from independent same-input rebuild')
        trace_cmd = [*record.command, '-Wl,--trace']
        trace_cmd[trace_cmd.index('-o') + 1] = str(verify / 'link-trace.so')
        trace = command(trace_cmd)
        (record.directory / 'link-trace.txt').write_text(trace.stdout + trace.stderr)
        for line in trace.stdout.splitlines():
            path = Path(line.strip())
            if not path.is_file():
                continue
            self._known_bytes(path)
        if checksum(verify / 'link-trace.so') != actual_hash:
            raise RuntimeMismatch('Native link trace changed compiled bytes')
        shutil.copyfile(output, record.directory / 'actual.so')
        record.sha256 = actual_hash
        record.artifacts['rebuild_sha256'] = checksum(rebuilt)
        self.check_inputs()

    def _cache_result(self, record, path):
        if path is None:
            return
        path = str(Path(path).resolve())
        if not Path(path).is_relative_to(self.directory / 'cache'):
            raise RuntimeMismatch('Generated native binary is outside the owned cache')
        prior = self.paths.get(path)
        expected = record.sha256 or (prior.sha256 if prior is not None and prior.source == record.source else None)
        if expected is None or checksum(path) != expected:
            raise RuntimeMismatch(f'Unrecorded or corrupted native cache binary: {path}')
        if record.sha256 is None:
            record.sha256 = expected
            record.artifacts['reused_build'] = prior.directory.name
        record.output = path
        self.paths[path] = record

    def observe(self, frame, event, arg):
        try:
            code = frame.f_code
            if same_code(code, self.driver.CudaLauncher.__init__.__code__):
                if event == 'call':
                    self._launcher(frame)
                elif event == 'return':
                    self.launchers.pop(id(frame), None)
            elif same_code(code, self.driver.compile_module_from_src.__code__):
                if event == 'call':
                    self._request(frame)
                elif event == 'return':
                    record = self.pending.pop(id(frame))
                    if arg is None or record.output != str(Path(arg.__file__).resolve()):
                        raise RuntimeMismatch('Native module did not load the verified binary')
                    self._cache_result(record, record.output)
                    record.artifacts['loaded_cuda_driver'] = [
                        self._known_bytes(path) for path in sorted({line.split()[-1] for line in
                        Path('/proc/self/maps').read_text().splitlines()
                        if len(line.split()) >= 6 and Path(line.split()[-1]).name.startswith('libcuda.so')})]
                    if not record.artifacts['loaded_cuda_driver']:
                        raise RuntimeMismatch('Compiled extension did not expose its loaded CUDA driver')
                    record.loaded = True
            elif code is self.builder._build.__code__ and id(frame.f_back) in self.pending:
                record = self.pending[id(frame.f_back)]
                if event == 'call':
                    self._build_start(frame, record)
                elif event == 'return':
                    self._build_end(frame, arg, record)
            elif (event == 'return' and code in (self.cache.FileCacheManager.get_file.__code__,
                                                self.cache.FileCacheManager.put.__code__)
                  and id(frame.f_back) in self.pending):
                self._cache_result(self.pending[id(frame.f_back)], arg)
        except Exception:
            self.failed = True
            raise

    def verified(self):
        self.check_inputs()
        if not self.complete or self.failed or self.pending:
            raise RuntimeMismatch('Incomplete native build verification')
        result = {}
        for record in self.records:
            if (record.directory / 'input.c').read_text() != record.source:
                raise RuntimeMismatch('Retained generated source changed')
            if 'reused_build' not in record.artifacts:
                for path in (record.directory / 'actual.so', record.directory / 'rebuild' /
                             (record.module + self.policy['extension_suffix'])):
                    if checksum(path) != record.sha256:
                        raise RuntimeMismatch('Retained native build evidence changed')
            if not record.loaded or not record.output or checksum(record.output) != record.sha256:
                raise RuntimeMismatch('Generated native binary changed after loading')
            result[(record.module, record.output)] = record.sha256
        return result

    def evidence(self):
        self.verified()
        return {'contract': CONTRACT, 'builds': [
            {'module': r.module, 'generator': r.generator, 'output': r.output, 'sha256': r.sha256,
             'source_sha256': checksum(r.directory / 'input.c'), 'command': r.command,
             'dependencies': r.dependencies, **r.artifacts} for r in self.records]}
