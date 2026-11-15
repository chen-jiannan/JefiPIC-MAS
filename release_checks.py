"""Portable integrity verification for the shared current source tree."""
from pathlib import Path, PurePosixPath
import hashlib,json
def verify_case(root,manifest):
    root=Path(root).resolve();data=json.loads((root/manifest).read_text())
    for row in data['files']:
        p=root/row['path'];p.resolve().relative_to(root)
        if p.stat().st_size!=row['bytes'] or hashlib.sha256(p.read_bytes()).hexdigest()!=row['sha256']:
            raise RuntimeError('Checksum mismatch: '+row['path'])
    return len(data['files'])
def verify_all(root):
    root=Path(root).resolve();count=0
    for line in (root/'SHA256SUMS.txt').read_text().splitlines():
        h,name=line.split('  ',1);rel=PurePosixPath(name)
        if rel.is_absolute() or '..' in rel.parts:raise RuntimeError('Invalid path')
        p=root.joinpath(*rel.parts)
        if hashlib.sha256(p.read_bytes()).hexdigest()!=h:raise RuntimeError('Checksum mismatch: '+name)
        count+=1
    for name in ['two_arm','sphere']:verify_case(root,'examples/'+name+'/revision_manifest.json')
    print('PASSED current repository checksums:',count,'files',flush=True)
    return count
