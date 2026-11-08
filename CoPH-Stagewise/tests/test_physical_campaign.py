import numpy as np
from coph_stagewise.physical_campaign import accounting,derivative_checks,overlap_checks


def test_held_accounting_and_quadrature():
    rows=accounting()
    assert len(rows)==45
    assert max(abs(r['residual']) for r in rows)<1e-12
    assert max(r['quadrature_work_error'] for r in rows)<1e-7


def test_full_potential_and_multistep_gradient():
    r=derivative_checks()
    assert r['max_force_error']<1e-7
    assert max(s['max_error'] for s in r['replay'])<1e-6
    assert r['context_change']['physical_jump']==0.
    assert r['context_change']['actuator_work']==0.


def test_supported_overlap_records_derivative_mismatch():
    r=overlap_checks()
    assert len(r['summary'])==4
    assert all(q['supported'] for q in r['profiles'])
    assert all(np.isfinite(q['max_gradient_gap']) and q['max_gradient_gap']>0 for q in r['summary'])
