"""Versioned repair of historical stage export; preserved inputs and fixed cadence."""
import argparse,ast,copy,hashlib,json,time
from pathlib import Path
from dataclasses import asdict
import torch
from .communication_repairs import *
from .communication_repair_suite import DEPENDENCIES,ROOT
from .receiver_corrections import load_saved_snapshot
from .evaluation import load,dump
from .attribution_audits import digest


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def codec_ast(path):
    tree=ast.parse(Path(path).read_text())
    return {n.name:hashlib.sha256(ast.dump(n,include_attributes=False).encode()).hexdigest() for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('encode_packet','decode_packet')}


def run(source,out):
    torch.set_num_threads(1);source=Path(source).resolve();out=Path(out).resolve();out.mkdir(parents=True,exist_ok=True);m=load(source/'manifest.json')
    helper=Path(__file__).parent/'communication_repairs.py';old=source/'source_archive/communication_repairs.py'
    assert codec_ast(helper)==codec_ast(old),'codec changed; cannot reuse matched codec benchmark'
    protocol=dict(version=2,initial_manifest_sha256=sha(source/'manifest.json'),input_snapshots={e['path']:e['sha256'] for e in m['inputs']},
        amendment='Keep original stage order; set copied LedgerExport rotation cursor to the selected stage. This preserves historical anchor selection.',
        failed_prior_rows_sha256=sha(source/'selection_rows.json'),codec_reused=sha(source/'matched_payload_rows.json'),codec_functions_ast=codec_ast(helper),
        sources={n:sha(ROOT/'coph_stagewise'/n) for n in DEPENDENCIES+('communication_selection_repair.py',)},
        expected_selection_pairs=108,expected_actual_receipts=24,no_new_scenes=True)
    p=out/'manifest.json'
    if p.exists():
        if load(p)!=protocol:raise ValueError('selection repair sources/inputs changed')
    else:
        with p.open('x') as f:json.dump(protocol,f,indent=2)
        (out/'source_archive').mkdir()
        for n in protocol['sources']:(out/'source_archive'/n).write_bytes((ROOT/'coph_stagewise'/n).read_bytes())
    selection=[];transport=[]
    for entry in m['inputs']:
        if sha(entry['path'])!=entry['sha256']:raise ValueError('snapshot changed')
        obj=load_saved_snapshot(entry['path'])
        if entry['meta']['group']=='acquisition':
            for sender in (0,1):
                for mode in ('rotation','receiver_relevant'):
                    s=copy.deepcopy(obj['agents'][sender]);a=copy.deepcopy(obj['agents'][1-sender]);x=obj['states'][sender];rx=obj['states'][1-sender].copy();before=digest([(st.readings,st.transitions) for st in a.stages]);physical=a.current_state
                    link=FixedRateRepairLink(CommunicationRepairOptions('compact_v1',mode));sent=link.offer(s,x,a,rx,8);received=link.deliver_due(a,rx);again=link.deliver_due(a,rx)
                    assert a.current_state==physical and digest([(st.readings,st.transitions) for st in a.stages])==before
                    transport.append(dict(entry['meta'],sender=sender,mode=mode,sent=sent,received=received,accounting=link.accounting,exactly_once=again['update'] is None))
        else:
            agent,x,packet,world,tick=obj
            if len(agent.stages)<2:continue
            route=tuple(r.q for r in agent.stages[0].readings if int(r.uid.split(':')[-1])%5)[:16]
            ad=PayloadAdvertisement(1,packet.sender,agent.agent_id,agent.now,agent.now+.96,0,route,(),())
            for mode in ('rotation','receiver_relevant'):
                a=copy.deepcopy(agent);p,metadata=select_payload(a,x,packet.sender,ad,mode)
                selection.append(dict(entry['meta'],request_kind='controlled oldest-retained measured route request',packet_stage=p.stage if p else None,
                    payload_bytes=len(encode_packet(p)) if p else 0,**metadata))
    assert len(selection)==216 and len(transport)==24
    dump(out/'selection_rows.json',selection);dump(out/'fixed_rate_transport_rows.json',transport)
    pairs=list(zip(selection[::2],selection[1::2]));eligible=[(a,b) for a,b in pairs if a['packet_stage'] is not None and b['packet_stage'] is not None];changed=sum(a['packet_stage']!=b['packet_stage'] for a,b in eligible)
    prior=load(source/'selection_rows.json');missing=sum(r['packet_stage'] is None for r in prior if r['mode']=='receiver_relevant')
    summary=dict(attempted_pairs=len(pairs),eligible_nonempty_pairs=len(eligible),changed_stage_pairs=changed,relevant_empty=sum(b['packet_stage'] is None for a,b in pairs),
        prior_relevant_empty=missing,actual_receipts=len(transport),normal_validation_eligible=sum(r['received']['events'][0]['reason']=='eligible' for r in transport),exactly_once=all(r['exactly_once'] for r in transport),
        codec_unchanged_and_reused=True,claim='controlled selection and actual receipt, no navigation benefit claim')
    dump(out/'summary.json',summary)
    (out/'REPORT.md').write_text('# Historical export repair\n\n'+json.dumps(summary,indent=2)+'\n\nThe original selection records are preserved. Their 108 changed outcomes included 72 empty payloads; that was not 108 successful selected packets. The repaired cursor preserves original ledger order and historical measurement anchors. Codec functions are AST-identical and the existing matched-evidence benchmark is reused explicitly. Actual navigation is evaluated separately in navigation_v2.\n')
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();print(run(a.source,a.out))
