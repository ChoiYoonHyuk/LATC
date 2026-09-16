import numpy as np
import pytest
from latc.affine import (fwht, regional_posterior, dense_regional_posterior,
                        posterior_prediction, augmented_rank, sample_target_array,
                        clean_labels)

@pytest.mark.parametrize('d',[1,2,3,6])
def test_fwht_is_unnormalized_involution(d):
    rng=np.random.default_rng(d)
    vector=rng.normal(size=(3,1 << d))
    np.testing.assert_allclose(fwht(fwht(vector)),(1 << d)*vector,atol=1e-12)

@pytest.mark.parametrize('d',[1,2,3,5])
@pytest.mark.parametrize('eta',[0,.01,.05,.2])
def test_transform_matches_independent_dense_dictionary(d,eta):
    rng=np.random.default_rng(100+d)
    for n in (0,min(4,1 << d),min(11,1 << d)):
        xs=rng.choice(1 << d,n,replace=False)
        ys=rng.integers(0,2,n)
        fast=regional_posterior(d,eta,xs,ys)
        dense=dense_regional_posterior(d,eta,xs,ys)
        np.testing.assert_allclose(fast.clean_sign,dense.clean_sign,atol=1e-12)
        if np.isfinite(fast.log_lr):
            assert fast.log_lr==pytest.approx(dense.log_lr,abs=1e-12)
        else:
            assert dense.log_lr==-np.inf

@pytest.mark.parametrize('eta',[0,.01,.1,.49])
def test_independent_rows_give_exact_likelihood_one(eta):
    qs=[0,1,2,4,8]
    p=regional_posterior(4,eta,qs,[1,0,1,1,0])
    assert p.rank==5 and p.log_lr==0.0


def test_inconsistent_zero_noise_posterior_has_no_reliable_mass():
    p=regional_posterior(3,0,[0,1,2,3],[0,0,0,1])
    assert p.log_lr==-np.inf
    assert not np.any(p.clean_sign)


def test_observed_targets_are_exact_even_under_noise():
    qs=([0,1,2,3],[4,7]);ys=([1,0,1,0],[1,1])
    pred,_,_=posterior_prediction(3,.2,qs,ys)
    for j in (0,1):
        np.testing.assert_array_equal(pred[j,qs[j]],ys[j])
    assert np.all((pred>=0)&(pred<=1))


def test_initial_predictions_and_prior():
    p,_,gamma=posterior_prediction(3,.05,([],[]),([],[]),prior=.7)
    np.testing.assert_array_equal(p,np.full((2,8),.5))
    np.testing.assert_allclose(gamma,[.7,.3])


def test_duplicate_and_invalid_queries_are_rejected():
    with pytest.raises(ValueError):regional_posterior(3,.1,[1,1],[0,1])
    with pytest.raises(ValueError):regional_posterior(3,.1,[8],[0])
    with pytest.raises(ValueError):regional_posterior(3,.1,[1],[2])
    with pytest.raises(ValueError):regional_posterior(3,.5,[1],[0])


def test_ranks_are_over_gf2_not_reals():
    assert augmented_rank([0,1,2,3])==3
    assert augmented_rank([0,1,2,4,8,15])==5


def test_target_law_and_seed_stability():
    a=sample_target_array(8,0,1234);b=sample_target_array(8,0,1234)
    np.testing.assert_array_equal(a.labels,b.labels)
    np.testing.assert_array_equal(a.labels[a.reliable_region],clean_labels(8,a.beta,a.slope))
    assert not a.labels.flags.writeable
