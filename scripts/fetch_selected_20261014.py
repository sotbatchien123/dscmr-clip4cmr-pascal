import urllib.request, zipfile, io, json, hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for name, repo, branch in [('DSCMR','penghu-cs/DSCMR','master'), ('CLIP4CMR','zhixiongz/CLIP4CMR','main')]:
    dest = ROOT/name/'upstream'
    if dest.exists():
        print(name, 'exists', flush=True)
        continue
    url = f'https://codeload.github.com/{repo}/zip/refs/heads/{branch}'
    raw = urllib.request.urlopen(url, timeout=90).read()
    z = zipfile.ZipFile(io.BytesIO(raw))
    assert z.testzip() is None
    for item in z.infolist():
        if item.is_dir(): continue
        target = dest.joinpath(*Path(item.filename).parts[1:]).resolve()
        if not target.is_relative_to(dest.resolve()): raise ValueError(item.filename)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(z.read(item))
    (dest.parent/'source.json').write_text(json.dumps({'repo':repo,'branch':branch,'archive_sha256':hashlib.sha256(raw).hexdigest()},indent=2))
    print(name, len(raw), 'bytes', len(z.infolist()), 'entries', flush=True)
