"""Fail-closed causal gate for frozen A3/A5, NEED-A4, and material-pH."""
import argparse, hashlib, json
from pathlib import Path
import numpy as np
import torch
from .algorithm import acquire_to_evidence, post_reading_branch
from .e2_dev_stack import acquisition_menu, choose
from .frozen_need_a4 import FrozenNeedA4
from .lookahead import LookaheadSpec, region_index
from .lookahead_bidirectional import BidirectionalLookaheadEnv, NeedPacket
from .material_executor import MaterialHamiltonianExecutor
from .value_model import VariableSetValueNet

HERE=Path(__file__).resolve().parent
CAMPAIGN=HERE/"results"/"material_campaign"
DEFAULT_A3_A5=CAMPAIGN/"a3_a5_frozen.pt"
DEFAULT_LOG=CAMPAIGN/"gate_2_full_stack.json"

def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def load_a3_a5(path=DEFAULT_A3_A5):
    path=Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"frozen A3/A5 checkpoint missing: {path}; run gate 1 first")
    training_log = CAMPAIGN / "a3_a5_training_log.json"
    if training_log.is_file():
        status = json.loads(training_log.read_text()).get("status", "")
        if not status.startswith("passed_"):
            raise RuntimeError(f"A3/A5 gate did not pass: {status}")
    payload=torch.load(path,map_location="cpu",weights_only=False)
    state=(payload.get("model_state_dict", payload.get("state_dict", payload))
           if isinstance(payload,dict) else payload)
    model=VariableSetValueNet(); model.load_state_dict(state,strict=True); model.eval()
    return model

def need_for(env,receiver,choice):
    region=region_index(tuple(choice.target_region))
    if region is None: raise ValueError("candidate is outside public look-ahead regions")
    belief=env.region_beliefs[receiver][(region,choice.modality)]
    return NeedPacket(region,choice.modality,float(np.sqrt(belief.variance)),1.,
                      (env.step_index+120)*env.config.dt,
                      float(env.positions[receiver][0]),1)

def run_gate(path=DEFAULT_A3_A5):
    model=load_a3_a5(path); a4=FrozenNeedA4(); trace=[]
    env=BidirectionalLookaheadEnv(LookaheadSpec(981337,surface=(.95,.06),traction=(.20,.90)))
    env.executor="material_ph"
    cand,choices,sets=acquisition_menu(env,"scout",8)
    index=choose(model,env,"scout",cand,sets,"acquire")
    selected=tuple(i for i in sets[index] if i>=0)
    trace.append({"stage":"A3","choice":list(sets[index]),"free":True})
    if not selected: raise RuntimeError("free A3 selected skip")
    state=env; packets=[]
    for i in selected:
        got=acquire_to_evidence(state,choices[i])
        if got is None: raise RuntimeError("A3 physical acquisition failed")
        state=got.env
        packets += [pid for pid,p in state.innovation_packets.items()
                    if p.provenance==got.evidence_id]
    trace.append({"stage":"evidence","packets":packets})
    rc,rchoices,rsets=acquisition_menu(state,"carrier",8)
    before_i=choose(model,state,"carrier",rc,rsets,"acquire")
    rselected=tuple(i for i in rsets[before_i] if i>=0)
    if not rselected: raise RuntimeError("free A5 selected skip before NEED")
    need=need_for(state,"carrier",rchoices[rselected[0]])
    nid=state.create_need("carrier",need)
    _,state=post_reading_branch(state,nid,True,continue_mission=False)
    if nid not in state.received["scout"]: raise RuntimeError("NEED delivery failed")
    trace.append({"stage":"NEED","region":need.region,"modality":need.modality})
    matching=[p for p in packets if state.innovation_packets[p].region==need.region
              and state.innovation_packets[p].modality==need.modality]
    if not matching: raise RuntimeError("free policies produced no matching innovation")
    pid=matching[0]; decision=a4.decide(state,"scout",pid,need)
    trace.append({"stage":"A4",**decision})
    if not decision["send"]: raise RuntimeError("frozen A4 held matching innovation")
    prior=state.clone(); _,delivered=post_reading_branch(state,pid,True,continue_mission=False)
    if pid not in delivered.received["carrier"]: raise RuntimeError("innovation delivery failed")
    rc2,_,rsets2=acquisition_menu(delivered,"carrier",8)
    after_i=choose(model,delivered,"carrier",rc2,rsets2,"acquire")
    changed=tuple(rsets[before_i])!=tuple(rsets2[after_i])
    trace.append({"stage":"A5","before":list(rsets[before_i]),"after":list(rsets2[after_i]),"changed":changed})
    if not changed: raise RuntimeError("memory did not change A5")
    executor=MaterialHamiltonianExecutor(); pos=prior.positions["carrier"]
    vel=prior.physics.world.agents[1].state.vel[0].detach().cpu().numpy(); goal=np.array((5.2,0.))
    f0,_=executor.proposal(pos,vel,goal,prior.observations()["carrier"])
    f1,_=executor.proposal(pos,vel,goal,delivered.observations()["carrier"])
    delta=float(np.linalg.norm(f1-f0)); trace.append({"stage":"belief_H_material_ph","force_delta":delta})
    if delta<=1e-7: raise RuntimeError("belief did not alter material-pH proposal")
    return {"status":"passed","scope":"engineering causal gate","trace":trace,
            "a3_a5_sha256":digest(path),"a4_sha256":digest(a4.checkpoint)}

def main():
    p=argparse.ArgumentParser(); p.add_argument("--a3-a5",type=Path,default=DEFAULT_A3_A5); p.add_argument("--output",type=Path,default=DEFAULT_LOG); a=p.parse_args()
    a.output.parent.mkdir(parents=True,exist_ok=True)
    try: report=run_gate(a.a3_a5)
    except Exception as e: report={"status":"pending_or_failed","error":str(e),"a3_a5_checkpoint":str(a.a3_a5.resolve())}
    a.output.write_text(json.dumps(report,indent=2)+"\n"); print(json.dumps(report,indent=2))
    if report["status"]!="passed": raise SystemExit(2)
if __name__=="__main__": main()
