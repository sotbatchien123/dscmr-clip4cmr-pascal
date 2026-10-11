"""Call the original Flickr30K-to-Arrow converter; preserve its preprocessing."""
import argparse
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--upstream', required=True)
    p.add_argument('--root', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    root = Path(a.root).resolve()
    for path in [root/'flickr30k-images', root/'karpathy/dataset_flickr30k.json']:
        if not path.exists():
            raise FileNotFoundError(path)
    sys.path.insert(0, str(Path(a.upstream).resolve()))
    from vilt.utils.write_f30k_karpathy import make_arrow
    make_arrow(str(root), str(Path(a.output).resolve()))


if __name__ == '__main__':
    main()
