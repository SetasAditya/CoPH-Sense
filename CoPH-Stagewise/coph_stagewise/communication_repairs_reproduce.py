"""Restore verified frozen communication sources for exact audit resumption."""
import argparse,hashlib,json,os,shutil,subprocess,sys,tempfile
from pathlib import Path


def restore(root,phase):
    root=Path(root).resolve();manifest=json.loads((root/'manifest.json').read_text());base=Path(tempfile.mkdtemp(prefix='coph_comm_frozen_'));package=base/'coph_stagewise';package.mkdir()
    for p in Path(__file__).parent.glob('*.py'):shutil.copyfile(p,package/p.name)
    for name,expected in manifest['sources'].items():
        p=root/'source_archive'/name
        if hashlib.sha256(p.read_bytes()).hexdigest()!=expected:raise ValueError('frozen source changed: '+name)
        shutil.copyfile(p,package/name)
    if phase=='consistency':
        m=json.loads((root/'consistency_v2/manifest.json').read_text())
        for name,expected in m['phase_sources'].items():
            p=root/'consistency_v2/source_archive'/name
            if hashlib.sha256(p.read_bytes()).hexdigest()!=expected:raise ValueError('phase source changed: '+name)
            shutil.copyfile(p,package/name)
    return base,manifest


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--phase',choices=['initial','consistency'],required=True);p.add_argument('--workers',type=int,default=8);a=p.parse_args();base,m=restore(a.out,a.phase)
    env=dict(os.environ,PYTHONPATH=str(base),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    if a.phase=='initial':
        code='import sys; from pathlib import Path; import coph_stagewise.communication_repair_suite as s; s.SOURCE=Path(sys.argv[1]); s.fault_suite(Path(sys.argv[2]),workers=int(sys.argv[3]))'
        cmd=[sys.executable,'-c',code,m['source'],str(a.out.resolve()),str(a.workers)]
    else:cmd=[sys.executable,'-m','coph_stagewise.communication_consistency_phase','--source',str(a.out.resolve()),'--out',str(a.out.resolve()/'consistency_v2'),'--workers',str(a.workers)]
    subprocess.run(cmd,env=env,check=True)


if __name__=='__main__':main()
