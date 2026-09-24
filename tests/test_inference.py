from itertools import product
import math
import numpy as np
import pytest
from latc.algebra import (affine_values, augmented_rank, dependency_basis, fwht,
                          likelihood_table, outcome_code, span_elements, syndrome_likelihood)
from latc.posterior import predict_two_regions, regional_posterior
from latc.verification import block_log_likelihood


@pytest.mark.parametrize("d",range(2,8))
@pytest.mark.parametrize("eta",[0,.05,.2])
def test_transform_parameter_dependency_and_syndrome_agree(d,eta):
    rng=np.random.default_rng(87+d)
    m=1<<d
    coordinates=rng.choice(m,size=min(m,d+3),replace=False)
    theta=int(rng.integers(2*m))
    labels=affine_values(d,theta)[coordinates] ^ (rng.random(len(coordinates))<eta).astype(np.uint8)
    posterior=regional_posterior(d,coordinates,labels,eta)
    clean=np.array([affine_values(d,t) for t in range(2*m)])
    errors=np.count_nonzero(clean[:,coordinates]!=labels,axis=1)
    weights=((errors==0).astype(float) if eta==0 else (1-eta)**(len(labels)-errors)*eta**errors)
    q=weights/weights.sum()
    means=q @ (1-2*clean.astype(int))
    ratio=2**len(labels)*weights.mean()
    np.testing.assert_allclose(posterior.sign_mean,means,atol=2e-12)
    assert math.exp(posterior.log_likelihood_ratio)==pytest.approx(ratio,abs=2e-12)
    assert likelihood_table(coordinates,eta)[outcome_code(labels)]==pytest.approx(ratio,abs=2e-12)
    assert syndrome_likelihood(coordinates,labels,eta)==pytest.approx(ratio,abs=2e-12)
    rank=augmented_rank(coordinates)
    assert np.mean(means**2)<=2**(rank-(d+1))+2e-12


def test_target_representation_identity():
    coordinates=[0,1,2,3,4]
    labels=[0,1,0,0,1]
    eta=.1
    posterior=regional_posterior(3,coordinates,labels,eta)
    numerator=np.zeros(8)
    for subset in range(1<<len(coordinates)):
        row=0
        sign=1
        for i,x in enumerate(coordinates):
            if subset>>i&1:
                row ^= 1|(x<<1)
                sign *= 1-2*labels[i]
        if row&1:
            numerator[row>>1] += (1-2*eta)**subset.bit_count()*sign
    np.testing.assert_allclose(posterior.sign_mean,numerator/math.exp(posterior.log_likelihood_ratio),atol=1e-12)


def test_fwht_involution_and_no_input_mutation():
    a=np.arange(32).reshape(4,8)
    original=a.copy()
    np.testing.assert_array_equal(fwht(fwht(a)),8*a)
    np.testing.assert_array_equal(a,original)
    with pytest.raises(ValueError):
        fwht(np.zeros(3))


def test_independent_rows_and_minimum_informative_parity():
    for coordinates in ([0],[0,1],[0,1,3],[0,1,2,4]):
        assert augmented_rank(coordinates)==len(coordinates)
        np.testing.assert_array_equal(likelihood_table(coordinates,.1),np.ones(1<<len(coordinates)))
    table=likelihood_table([0,1,2,3],.1)
    testing_error=.5-.25*np.mean(np.abs(table-1))
    assert testing_error==pytest.approx(.5-.8**4/4)
    assert sorted(v.bit_count() for v in span_elements(dependency_basis(range(8))))==[0]+[4]*14+[8]


def test_zero_likelihood_and_persistent_observed_targets():
    prediction=predict_two_regions(3,[[0,1,2,3],[]],[[0,0,0,1],[]],0)
    np.testing.assert_array_equal(prediction.gamma,[0,1])
    np.testing.assert_array_equal(prediction.probabilities[0,:4],[0,0,0,1])
    assert prediction.regional[0].log_likelihood_ratio==-math.inf
    with pytest.raises(ValueError,match="zero probability"):
        predict_two_regions(3,[[0,1,2,3],[0,1,2,3]],[[0,0,0,1],[0,0,0,1]],0)


def test_target_noise_prior_and_normalizations():
    q=([0,1,2],[4,5])
    y=([0,0,0],[0,1])
    prediction=predict_two_regions(3,q,y,.1,.7)
    # Target x=3 is represented by three acquired basis rows, plus its OWN noise.
    assert prediction.probabilities[0,3]==pytest.approx(.5-.7*.8**4/2)
    assert prediction.unqueried_risk==pytest.approx(16/11*prediction.population_risk)
    targets=([3,6,7],[0,1,2,3,6,7])
    decomposition=prediction.decomposition(targets)
    assert decomposition['conditional_brier']==pytest.approx(sum(decomposition[k] for k in ['residual','parameter_uncertainty','indicator_uncertainty']))
    assert all(decomposition[k]>=-1e-13 for k in decomposition)
    with pytest.raises(ValueError,match="unqueried"):
        prediction.common_risk(([0],[0]))


@pytest.mark.parametrize("d",[3,4,5])
def test_block_symmetry_and_likelihood(d):
    rng=np.random.default_rng(d)
    y=rng.integers(0,2,size=(10,1<<d),dtype=np.uint8)
    transform=block_log_likelihood(y,.1)
    for i,row in enumerate(y):
        posterior=regional_posterior(d,range(1<<d),row,.1)
        assert transform[i]==pytest.approx(posterior.log_likelihood_ratio,abs=1e-12)
    restriction=affine_values(d,7)
    np.testing.assert_allclose(block_log_likelihood(y ^ restriction,.1),transform,atol=1e-12)


@pytest.mark.parametrize("coordinates,labels,eta",[([0,0],[0,1],.1),([8],[0],.1),([0],[2],.1),([0],[0],.5),([0],[],.1)])
def test_invalid_histories_rejected(coordinates,labels,eta):
    with pytest.raises(ValueError):
        regional_posterior(3,coordinates,labels,eta)
