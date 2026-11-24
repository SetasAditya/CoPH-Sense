"""Validate local gallery resources, media provenance and optional trusted logs."""
from pathlib import Path
from html.parser import HTMLParser
import argparse,hashlib,json,subprocess,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];SITE=ROOT/'project-page/dist'
NAMES=['dynamics','hazards','reciprocity','encounters','uncertainty']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def sha_json(v):return hashlib.sha256(json.dumps(v,sort_keys=True).encode()).hexdigest()
class Links(HTMLParser):
    def __init__(self):super().__init__();self.links=[]
    def handle_starttag(self,tag,attrs):
        for k,v in attrs:
            if k in ['href','src','poster'] and v and not v.startswith(('#','http:','https:','mailto:')):self.links.append(v.split('#')[0])
def verify(archive=None):
    meta=json.loads((SITE/'assets/provenance.json').read_text());assert set(meta['frames'])==set(NAMES)
    checked=[];counts={}
    for name in NAMES:
        for ext in ['mp4','gif','png']:
            p=SITE/'assets'/f'{name}.{ext}';assert p.is_file() and sha(p)==meta['artifacts'][p.name]['sha256'];checked.append(str(p.relative_to(ROOT)))
        frames=json.loads((SITE/'assets'/f'{name}.frames.json').read_text());assert frames==meta['frames'][name]
        probe=json.loads(subprocess.check_output(['/usr/bin/ffprobe' if Path('/usr/bin/ffprobe').exists() else 'ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=width,height,nb_frames,codec_name','-of','json',str(SITE/'assets'/f'{name}.mp4')]))['streams'][0]
        assert int(probe['nb_frames'])==len(frames) and probe['codec_name']=='h264' and (probe['width'],probe['height'])==(1280,800)
        assert all(f['video_frame']==i and abs(f['video_seconds']-i/10)<1e-12 for i,f in enumerate(frames))
        if name=='uncertainty':assert all(f['physical_time'] is None for f in frames)
        else:assert all(a['mission_seconds']<=b['mission_seconds'] for a,b in zip(frames,frames[1:]))
        counts[name]=len(frames)
    h=Links();h.feed((SITE/'index.html').read_text());assert all((SITE/p).is_file() for p in h.links),h.links
    assert meta['renderer_sha256']==sha(ROOT/'CoPH-Stagewise/coph_stagewise/scenario_presentation.py')
    assert meta['extractor_sha256']==sha(ROOT/'CoPH-Stagewise/coph_stagewise/scenario_logs.py')
    paper=ROOT.parent/'overleaf-coph-paper/sections/scenarios.tex'
    if paper.exists():
        text=paper.read_text();assert text.count('3D PLACEHOLDER:')==2
        for label in ['fig:discovery3d','fig:recovery3d']:assert '\\label{'+label+'}' in text
    result={'media_files':len(checked),'frame_counts':counts,'local_links':len(h.links),'source_hashes_verified':True,'paper_placeholders':2 if paper.exists() else 'manuscript unavailable','new_training_or_navigation_runs':False}
    if archive is not None:
        archive=Path(archive).resolve();sources={}
        for inputs in meta['sources'].values():
            for source in inputs:
                if 'path' in source:sources[source['path']]=source['sha256'];assert sha(archive/source['path'])==source['sha256'],source['path']
        sys.path.insert(0,str(ROOT/'CoPH-Stagewise'))
        from coph_stagewise import scenario_logs as logs
        from coph_stagewise.maze_visuals import landscape
        from coph_stagewise.records import Config
        import torch
        torch.set_num_threads(1);errors=[];snapshots=0
        for name,loader,key,who in [('hazards',logs.load_hazards,'field',1),('reciprocity',logs.load_crossing,'reciprocal_fixed',0)]:
            data=loader(archive);ep=data['episodes'][key];cfg=Config(**ep['config'])
            for f in meta['frames'][name]:
                fr=logs.stagewise_frame(ep,f['mission_seconds'])
                for a in fr['agents']:
                    v=next(v for v in f['agent_records'] if v['agent']==a['agent']);assert v['trajectory_index']==a.get('trajectory_index') and v['model_version']==a.get('model_version');snapshots+=1
                if f['surface']:
                    assert f['surface']['finite'];assert f['surface']['max_supported_H'] is None or f['surface']['max_supported_H']<=12
                    assert sha_json(fr['agents'][who]['model'])==f['surface']['model_sha256']
            for seq in ep['trajectories']:
                for r in seq[::20]:
                    q=np.array(r['x'][:2]);v=np.array(r['x'][2:]);H,F,mask=landscape(r['control_model'],q[None],v,cfg);errors.append(float(np.linalg.norm(F[0]-1.4*v-np.array(r['nominal']))))
            for e in data['episodes'].values():
                for agent in e['agents']:assert abs(sum(agent['components'].values())-agent['cost'])<1e-8
        assert max(errors)<1e-8,max(errors)
        native=logs.load_drag(archive);online=native['arms']['coph_online_static_protected']['arrays']['positions']
        for f in meta['frames']['dynamics']:assert f['logged_step']*.1==f['mission_seconds'] and hashlib.sha256(online[f['logged_step']].tobytes()).hexdigest()==f['state_sha256']
        encounter=logs.load_encounter(archive)
        for f in meta['frames']['encounters']:assert f['mission_seconds']==encounter['times'][f['logged_sample']]
        assert len(logs.regenerate_uncertainty_geometry(archive,12)['queries'])==12
        assert all(sha(archive/p)==digest for p,digest in sources.items())
        result.update(archive_sources=len(sources),exact_model_snapshots=snapshots,force_checks=len(errors),max_force_error=max(errors),regenerated_query_pairs=12,original_inputs_unchanged=True)
    return result
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--archive',type=Path);args=ap.parse_args();result=verify(args.archive)
    (ROOT/'docs/presentation').mkdir(parents=True,exist_ok=True);(ROOT/'docs/presentation'/('media_checks.json' if args.archive else 'static_checks.json')).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
