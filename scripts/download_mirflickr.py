"""Download original MIRFlickr archives from links on the official page.
Does not create paper-specific features/splits or download the 1M collection.
"""
import argparse
import hashlib
import json
import urllib.parse
import urllib.request
import zipfile
from html.parser import HTMLParser
from pathlib import Path

PAGE = 'https://press.liacs.nl/mirflickr/mirdownload.html'


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []

    def handle_starttag(self, tag, attrs):
        if tag == 'a':
            href = dict(attrs).get('href')
            if href:
                self.links.append(urllib.parse.urljoin(PAGE, href))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', required=True)
    a = p.parse_args()
    out = Path(a.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(PAGE, timeout=30) as r:
        html = r.read().decode('utf-8', errors='replace')
    parser = Links()
    parser.feed(html)
    manifest = []
    for name in ['mirflickr25k.zip', 'mirflickr25k_annotations_v080.zip']:
        matches = [u for u in parser.links if Path(urllib.parse.urlparse(u).path).name == name]
        if not matches:
            raise RuntimeError(f'Official page no longer links {name}; inspect it manually')
        url = matches[0]
        final = out/name
        if final.exists():
            raise FileExistsError(f'{final}: refusing to replace an existing download')
        partial = out/(name+'.partial')
        sha, md5 = hashlib.sha256(), hashlib.md5()
        size = 0
        with urllib.request.urlopen(url, timeout=60) as r, partial.open('wb') as f:
            expected_size = r.headers.get('Content-Length')
            while True:
                block = r.read(1024*1024)
                if not block:
                    break
                f.write(block)
                sha.update(block)
                md5.update(block)
                size += len(block)
            if expected_size and int(expected_size) != size:
                raise RuntimeError('Incomplete HTTP transfer')
        with zipfile.ZipFile(partial) as archive:
            bad = archive.testzip()
            if bad:
                raise RuntimeError(f'Corrupt archive member: {bad}')
        if name == 'mirflickr25k.zip' and md5.hexdigest().lower() != 'a23d0a8564ee84cda5622a6c2f947785':
            raise RuntimeError('MD5 differs from the official page checked 2026-10-11; verify release before using')
        partial.rename(final)
        manifest.append({'file': name, 'source': url, 'bytes': size, 'sha256': sha.hexdigest(), 'md5': md5.hexdigest()})
        (out/'downloads.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
        print(f'Validated {name}: {size} bytes')


if __name__ == '__main__':
    main()
