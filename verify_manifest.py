"""Check SHA-256 for every released file; standard library only."""
from pathlib import Path
import hashlib
import sys

def main():
    root=Path(__file__).resolve().parent
    errors=[]; count=0
    for line in (root/'MANIFEST.sha256').read_text(encoding='utf-8').splitlines():
        expected,relative=line.split('  ',1)
        p=root/relative
        h=hashlib.sha256()
        if not p.is_file(): errors.append(relative+' missing'); continue
        with p.open('rb') as f:
            while block:=f.read(1024*1024): h.update(block)
        if h.hexdigest()!=expected: errors.append(relative+' changed')
        count+=1
    print(f'Checked {count} files; {len(errors)} errors.')
    for e in errors: print(e)
    return 1 if errors else 0

if __name__=='__main__':
    sys.exit(main())
