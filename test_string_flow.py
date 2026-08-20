import torch
from rankflow import mm_tensor
from geometry_flow import residual_vector, naive_theta
from string_flow import strassen8_theta, canonicalize, align_endpoint, interpolate_string, reparameterize

def test_strassen_endpoint_exact():
    T=mm_tensor(2)
    q=canonicalize(strassen8_theta())
    assert float(residual_vector(q,T,2,8).norm()) < 1e-12

def test_alignment_preserves_endpoint():
    T=mm_tensor(2)
    s=canonicalize(naive_theta(2,"cpu"))
    e,_=align_endpoint(s,canonicalize(strassen8_theta()))
    assert float(residual_vector(e,T,2,8).norm()) < 1e-12

def test_reparameterize_keeps_endpoints():
    s=canonicalize(naive_theta(2,"cpu")); e,_=align_endpoint(s,canonicalize(strassen8_theta()))
    p=interpolate_string(s,e,11); q=reparameterize(p)
    assert torch.allclose(p[0],q[0]); assert torch.allclose(p[-1],q[-1])
