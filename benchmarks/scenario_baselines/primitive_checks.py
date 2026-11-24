"""Independent equation-level checks, never a native reproduction or comparison.

DistGP: GP conditional/FITC factors and Gaussian belief propagation, paper (2)--(5),
(8)--(10). Exposure planning: log-normal moments, risk (3), route objective (4)--(5).
The tiny inputs are ours and are frozen separately; no published endpoint is tested.
"""
from __future__ import annotations
import argparse
import hashlib
import heapq
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def kernel(x, z, variance, lengthscale):
    x = np.asarray(x, dtype=np.float64).reshape(-1, 1)
    z = np.asarray(z, dtype=np.float64).reshape(-1, 1)
    return variance * np.exp(-.5 * ((x-z.T)/lengthscale)**2)


def gaussian_checks(cfg):
    """FITC diagonal residuals, conditional GP tree, exact GBP on that tree."""
    n = len(cfg['blocks']); m = len(cfg['blocks'][0]['inducing'])
    eye = np.eye(m); par = cfg['kernel']; jitter = cfg['jitter']
    K = lambda a,b: kernel(a,b,par['variance'],par['lengthscale'])
    Z = [np.array(b['inducing']) for b in cfg['blocks']]
    Ks = [K(z,z)+jitter*eye for z in Z]
    unary_J = []; unary_h = []; fitc_errors=[]
    for b,z,kzz in zip(cfg['blocks'],Z,Ks):
        x = np.array(b['x']); y = np.array(b['y'])
        A = np.linalg.solve(kzz,K(z,x)).T
        residual = par['variance']-np.einsum('ij,ji->i',A,K(z,x)) + cfg['noise_std']**2
        if not np.all(residual>0): raise ValueError('Nonpositive FITC residual')
        J = A.T @ (A/residual[:,None]); h = A.T @ (y/residual)
        unary_J.append(J); unary_h.append(h)
        # Independent direct Gaussian conditioning of the same FITC observation model.
        obs_cov = A@kzz@A.T+np.diag(residual)
        direct_mean = kzz@A.T@np.linalg.solve(obs_cov,y)
        direct_cov = kzz-kzz@A.T@np.linalg.solve(obs_cov,A@kzz)
        canonical_J = np.linalg.inv(kzz)+J
        fitc_errors.append(dict(mean=float(np.max(np.abs(np.linalg.solve(canonical_J,h)-direct_mean))),
                               covariance=float(np.max(np.abs(np.linalg.inv(canonical_J)-direct_cov)))))
    unary_J[0] = unary_J[0]+np.linalg.inv(Ks[0])
    edges = {}; neighbors=[[] for _ in range(n)]
    for parent,child in cfg['tree_edges']:
        A = np.linalg.solve(Ks[parent],K(Z[parent],Z[child])).T
        S = Ks[child]-A@K(Z[parent],Z[child])
        Sinv = np.linalg.inv(S)
        J = np.block([[A.T@Sinv@A,-A.T@Sinv],[-Sinv@A,Sinv]])
        edges[parent,child]=J
        edges[child,parent]=np.block([[J[m:,m:],J[m:,:m]],[J[:m,m:],J[:m,:m]]])
        neighbors[parent].append(child);neighbors[child].append(parent)
    dense_J=np.zeros((n*m,n*m));dense_h=np.concatenate(unary_h)
    for i in range(n):dense_J[i*m:(i+1)*m,i*m:(i+1)*m]+=unary_J[i]
    for i,j in cfg['tree_edges']:
        ix=np.r_[np.arange(i*m,(i+1)*m),np.arange(j*m,(j+1)*m)]
        dense_J[np.ix_(ix,ix)]+=edges[i,j]
    dense_mean=np.linalg.solve(dense_J,dense_h);dense_cov=np.linalg.inv(dense_J)
    messages={(i,j):(np.zeros((m,m)),np.zeros(m)) for i,j in edges}
    converged=False;changes=[]
    for iteration in range(cfg['max_sweeps']):
        new={};change=0.
        for (i,j),pair in edges.items():
            Ji=unary_J[i].copy();hi=unary_h[i].copy()
            for k in neighbors[i]:
                if k!=j:Ji+=messages[k,i][0];hi+=messages[k,i][1]
            Ji+=pair[:m,:m]
            outJ=pair[m:,m:]-pair[m:,:m]@np.linalg.solve(Ji,pair[:m,m:])
            outh=-pair[m:,:m]@np.linalg.solve(Ji,hi)
            outJ=(outJ+outJ.T)/2
            change=max(change,float(np.max(np.abs(outJ-messages[i,j][0]))),float(np.max(np.abs(outh-messages[i,j][1]))))
            new[i,j]=(outJ,outh)
        messages=new;changes.append(change)
        if change<=cfg['message_tolerance']:converged=True;break
    means=[];covs=[]
    for i in range(n):
        J=unary_J[i].copy();h=unary_h[i].copy()
        for j in neighbors[i]:J+=messages[j,i][0];h+=messages[j,i][1]
        means.append(np.linalg.solve(J,h));covs.append(np.linalg.inv(J))
    mean_error=float(np.max(np.abs(np.concatenate(means)-dense_mean)))
    cov_error=max(float(np.max(np.abs(c-dense_cov[i*m:(i+1)*m,i*m:(i+1)*m]))) for i,c in enumerate(covs))
    passed=converged and mean_error<=cfg['check_tolerance'] and cov_error<=cfg['check_tolerance'] and all(max(r.values())<=cfg['check_tolerance'] for r in fitc_errors)
    return dict(passed=passed,kind='independent_primitive_validation_only',native_reproduction=False,
                fitc_direct_conditioning_errors=fitc_errors,tree_gbp_mean_max_error=mean_error,
                tree_gbp_marginal_covariance_max_error=cov_error,sweeps=len(changes),
                message_changes=changes,converged=converged,comparison_target='direct solve of the identical approximate Gaussian factor graph, not a full-GP prior or published metric')


def moments(mu,sigma):
    mean=np.exp(mu+.5*sigma*sigma)
    var=np.expm1(sigma*sigma)*np.exp(2*mu+sigma*sigma)
    return mean,var


def shortest(cfg,risk,heuristic):
    speed=np.asarray(cfg['speed_fraction'],dtype=float);rows,cols=speed.shape
    start=tuple(cfg['start']);goal=tuple(cfg['goal']);density=(1+cfg['beta']*risk)/(cfg['vmax']*np.where(speed>0,speed,1))
    minimum=float(density[speed>0].min())
    def h(p):return float(np.linalg.norm(np.array(p)-goal)*cfg['cell_size']*minimum) if heuristic else 0.
    distance={start:0.};parents={};queue=[(h(start),0.,start)];expanded=0
    while queue:
        _,g,p=heapq.heappop(queue)
        if g!=distance[p]:continue
        expanded+=1
        if p==goal:break
        for dr,dc in cfg['neighbor_order']:
            q=(p[0]+dr,p[1]+dc)
            if not (0<=q[0]<rows and 0<=q[1]<cols) or speed[q]<=0:continue
            if dr and dc and (speed[p[0]+dr,p[1]]<=0 or speed[p[0],p[1]+dc]<=0):continue
            cost=cfg['cell_size']*np.hypot(dr,dc)*density[p]
            proposed=g+cost
            if proposed<distance.get(q,float('inf')):
                distance[q]=proposed;parents[q]=p;heapq.heappush(queue,(proposed+h(q),proposed,q))
    if goal not in distance:raise ValueError('Frozen map is disconnected')
    route=[goal]
    while route[-1]!=start:route.append(parents[route[-1]])
    route=route[::-1];T=0.;E=0.
    for a,b in zip(route[:-1],route[1:]):
        dt=cfg['cell_size']*np.linalg.norm(np.array(b)-a)/(cfg['vmax']*speed[a])
        T+=dt;E+=risk[a]*dt
    return dict(path=[list(x) for x in route],objective=float(distance[goal]),time=float(T),exposure=float(E),expanded=expanded,component_sum=float(T+cfg['beta']*E))


def exposure_checks(cfg):
    mu=np.array(cfg['log_mean'],dtype=float);sigma=np.array(cfg['log_std'],dtype=float)
    mean,var=moments(mu,sigma)
    nodes,weights=np.polynomial.hermite.hermgauss(cfg['quadrature_order'])
    samples=np.exp(mu[...,None]+np.sqrt(2)*sigma[...,None]*nodes)
    qmean=np.sum(samples*weights,axis=-1)/np.sqrt(np.pi)
    qvar=np.sum((samples-qmean[...,None])**2*weights,axis=-1)/np.sqrt(np.pi)
    error=max(float(np.max(np.abs(qmean-mean))),float(np.max(np.abs(qvar-var))))
    risk=np.array(cfg['threat_multiplier'])*(mean+cfg['alpha']*np.sqrt(var))
    a=shortest(cfg,risk,True);d=shortest(cfg,risk,False)
    gate=error<=cfg['check_tolerance'] and abs(a['objective']-d['objective'])<=cfg['check_tolerance'] and abs(a['objective']-a['component_sum'])<=cfg['check_tolerance']
    return dict(passed=bool(gate),kind='independent_primitive_validation_only',native_reproduction=False,
                moments_quadrature_max_error=error,astar=a,dijkstra=d,
                objective_agreement=float(abs(a['objective']-d['objective'])),
                comparison_target='same fixed 8-connected objective with an admissible minimum-density heuristic; not the published IPP/acquisition loop')


def run():
    manifest=json.loads((ROOT/'primitive_manifest.json').read_text())
    for name,expected in manifest['sha256'].items():
        if sha(ROOT/name)!=expected:raise ValueError('Frozen primitive input/source changed: '+name)
    cfg=json.loads((ROOT/'primitive_inputs.json').read_text())
    result=dict(kind='independent_primitive_validation_only',native_reproduction=False,sota_comparison=False,
                manifest_sha256=sha(ROOT/'primitive_manifest.json'),distgp=gaussian_checks(cfg['distgp']),
                exposure=exposure_checks(cfg['exposure']))
    result['passed']=result['distgp']['passed'] and result['exposure']['passed']
    return result


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
    result=run()
    if args.out.exists():
        if json.loads(args.out.read_text())!=result:raise ValueError('Refuse differing existing check outputs')
    else:args.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
    if not result['passed']:raise SystemExit(1)


if __name__=='__main__':main()
