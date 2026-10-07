// Local path wiring: config/local-paths.json resolves against the share root,
// from any cwd, identically in the PowerShell and Python consumers. Nothing here
// needs the downloaded runtime to exist -- Setup.bat creates it later.
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const os = require('node:os');
const { spawnSync } = require('node:child_process');
const root = path.resolve(__dirname, '..');
const config = JSON.parse(fs.readFileSync(path.join(root, 'config/local-paths.json'), 'utf8'));
const keys = Object.keys(config);
const ps = path.join(process.env.SystemRoot, 'System32/WindowsPowerShell/v1.0/powershell.exe');
const quote = s => "'" + s.replaceAll("'", "''") + "'";

function run(exe, args, env = {}) {
  return spawnSync(exe, args, { cwd: os.tmpdir(), encoding: 'utf8', windowsHide: true,
    env: { ...process.env, PYTHONDONTWRITEBYTECODE: '1', PYTHONUTF8: '1', ...env } });
}
function ok(result) {
  assert.equal(result.status, 0, `${result.error || ''}\n${result.stdout}\n${result.stderr}`);
  return result.stdout.trim();
}
// `dir` is the tree whose scripts are dot-sourced; `runtime` also loads koharu-runtime.ps1.
function powershell(code, { dir = root, runtime = false, env } = {}) {
  const load = ['local-paths', ...(runtime ? ['koharu-runtime'] : [])]
    .map(name => `. ${quote(path.join(dir, `scripts/${name}.ps1`))}; `).join('');
  return run(ps, ['-NoProfile', '-ExecutionPolicy', 'Bypass', '-Command',
    `$ErrorActionPreference='Stop'; ${load}${code}`], env);
}

// The setup-installed interpreter when it exists, else whatever is on PATH.
const setupPython = config.python_ocr && path.resolve(root, config.python_ocr);
const python = setupPython && fs.existsSync(setupPython) ? setupPython : 'python';
const noPython = run(python, ['--version']).status !== 0 && 'no python on PATH and setup has not run';
const pyImportFrom = dir => `import sys; sys.path.insert(0, ${JSON.stringify(path.join(dir, 'scripts'))}); from local_paths import dependency_path; `;
const pyImport = pyImportFrom(root);

test('every configured path is relative, so the tree works from any install folder', () => {
  assert.ok(keys.length > 0, 'config/local-paths.json is empty');
  for (const [key, value] of Object.entries(config)) {
    assert.equal(typeof value, 'string', key);
    assert.ok(!path.isAbsolute(value), `${key} is absolute: ${value}`);
  }
});

test('PowerShell resolves every key against the share root from a foreign cwd', () => {
  const resolved = JSON.parse(ok(powershell(`$r=@{}; foreach ($k in @(${keys.map(quote).join(',')})) {$r[$k]=Get-BireLatePath $k}; $r | ConvertTo-Json -Compress`)));
  for (const [key, value] of Object.entries(config)) assert.equal(resolved[key], path.resolve(root, value), key);
});

test('Python resolves every key against the share root from a foreign cwd', { skip: noPython }, () => {
  const resolved = JSON.parse(ok(run(python, ['-B', '-c', pyImport +
    `import json; print(json.dumps({k:str(dependency_path(k)) for k in ${JSON.stringify(keys)}}))`])));
  for (const [key, value] of Object.entries(config)) assert.equal(resolved[key], path.resolve(root, value), key);
});

test('a misspelled dependency fails explicitly in both consumers', () => {
  const results = [powershell('Get-BireLatePath missing_dependency')];
  if (!noPython) results.push(run(python, ['-B', '-c', pyImport + 'dependency_path("missing_dependency")']));
  for (const result of results) {
    assert.notEqual(result.status, 0);
    assert.match(result.stderr, /Unknown or empty BireLate path/);
  }
});

// The recipient's install folder may contain spaces. Copy only the files the
// resolvers and the PATH function read into a temp root whose name has one.
function spacedRoot(t) {
  const dir = fs.realpathSync.native(fs.mkdtempSync(path.join(os.tmpdir(), 'bire late ')));
  t.after(() => fs.rmSync(dir, { recursive: true, force: true }));
  for (const file of ['scripts/local-paths.ps1', 'scripts/koharu-runtime.ps1', 'scripts/local_paths.py', 'config/local-paths.json']) {
    fs.mkdirSync(path.dirname(path.join(dir, file)), { recursive: true });
    fs.copyFileSync(path.join(root, file), path.join(dir, file));
  }
  return dir;
}

test('both consumers resolve every key inside an install root whose name has a space', (t) => {
  const dir = spacedRoot(t);
  assert.match(dir, / /);
  const ps5 = JSON.parse(ok(powershell(`$r=@{}; foreach ($k in @(${keys.map(quote).join(',')})) {$r[$k]=Get-BireLatePath $k}; $r | ConvertTo-Json -Compress`, { dir })));
  for (const [key, value] of Object.entries(config)) assert.equal(ps5[key], path.resolve(dir, value), key);
  if (noPython) return;
  const py = JSON.parse(ok(run(python, ['-B', '-c', pyImportFrom(dir) +
    `import json; print(json.dumps({k:str(dependency_path(k)) for k in ${JSON.stringify(keys)}}))`])));
  for (const [key, value] of Object.entries(config)) assert.equal(py[key], path.resolve(dir, value), key);
});

// cudart64_13.dll is found BY NAME through PATH, and the server panics at start
// without it. A fake %LOCALAPPDATA% with a planted installed-app runtime proves
// which folder the function really chooses.
function runtimePathRun(t, withBootstrap, code) {
  const dir = spacedRoot(t);
  const bootstrap = path.join(dir, config.cuda_bootstrap);
  if (withBootstrap) fs.mkdirSync(bootstrap, { recursive: true });
  const appData = path.join(dir, 'appdata');
  fs.mkdirSync(path.join(appData, 'koharu/runtime/llama.cpp/b9999/windows-cuda13-x64'), { recursive: true });
  const out = ok(powershell(code, { dir, runtime: true, env: { LOCALAPPDATA: appData } }));
  return { bootstrap, appData, out };
}
const marked = (out, tag) => out.split(/\r?\n/).filter(line => line.startsWith(tag)).at(-1).slice(tag.length);

test('Add-KoharuRuntimePath puts the local CUDA bootstrap first on PATH, in a root with a space', (t) => {
  const { bootstrap, out } = runtimePathRun(t, true,
    `Add-KoharuRuntimePath | Out-Null; 'FIRST=' + ($env:PATH -split ';')[0]`);
  assert.equal(marked(out, 'FIRST='), bootstrap);
});

test('without a local CUDA bootstrap, Add-KoharuRuntimePath puts no installed-app koharu folder on PATH', (t) => {
  const { appData, out } = runtimePathRun(t, false,
    `$before = @($env:PATH -split ';'); Add-KoharuRuntimePath | Out-Null; ` +
    `$added = @($env:PATH -split ';' | Where-Object { $_ -and ($before -notcontains $_) }); ` +
    `'ADDED=' + (ConvertTo-Json -InputObject $added -Compress)`);
  const koharu = path.join(appData, 'koharu').toLowerCase();
  const added = JSON.parse(marked(out, 'ADDED='));
  assert.deepEqual(added.filter(entry => entry.toLowerCase().startsWith(koharu)), []);
});
