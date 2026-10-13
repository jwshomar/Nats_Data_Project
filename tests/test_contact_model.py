"""Numerical, prediction, and split guarantees for the contact model."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import check_grad
from scipy.special import expit

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from contact_model import (
    ContactModel,
    batter_projection,
    objective_gradient,
    pair_sensitivity_split,
    prepare_swings,
    reference_contexts,
    split_swings,
)


@pytest.fixture
def swings():
    rng = np.random.default_rng(123)
    n = 4000
    batter = rng.integers(1, 13, n)
    pitcher = rng.integers(1, 13, n)
    frame = pd.DataFrame(
        {
            "batter_id": batter,
            "pitcher_id": pitcher,
            "bat_side": np.where(batter <= 6, "L", "R"),
            "throw_side": np.where(pitcher <= 6, "L", "R"),
            "is_fb": rng.integers(0, 2, n),
            "is_in_zone": rng.integers(0, 2, n),
            "is_swing": 1,
        }
    )
    eta = 0.5 + 0.5 * frame.is_in_zone + (batter - 6.5) / 6 - (pitcher - 6.5) / 6
    frame["is_miss"] = 1 - rng.binomial(1, expit(eta))
    return prepare_swings(frame)


def test_gradient_and_stable_extreme_logits(swings):
    model = ContactModel().fit(swings)
    design = model.design_matrix(swings)
    theta = np.random.default_rng(1).normal(size=model.n_parameters)
    y = swings.contact.to_numpy()
    error = check_grad(
        lambda t: objective_gradient(t, design, y, model.precision)[0],
        lambda t: objective_gradient(t, design, y, model.precision)[1],
        theta,
    )
    assert error / np.sqrt(model.n_parameters) < 5e-4
    loss, gradient = objective_gradient(theta * 1000, design, y, model.precision)
    assert np.isfinite(loss) and np.isfinite(gradient).all()


def test_split_disjoint_covered_reproducible_and_target_independent(swings):
    parts = split_swings(swings)
    flipped = swings.assign(contact=1 - swings.contact, is_miss=swings.contact)
    other = split_swings(flipped)
    indices = [set(part.index) for part in parts[:3]]
    assert not indices[0] & indices[1]
    assert not indices[0] & indices[2]
    assert not indices[1] & indices[2]
    assert set.union(*indices) == set(swings.index)
    for left, right in zip(parts[:3], other[:3]):
        assert left.index.equals(right.index)
        for player in ["batter_id", "pitcher_id"]:
            assert set(left[player]) == set(swings[player])
    dev = pd.concat(parts[:2])
    left, right = pair_sensitivity_split(dev)
    keys = ["batter_id", "pitcher_id"]
    assert not set(map(tuple, left[keys].to_numpy())) & set(
        map(tuple, right[keys].to_numpy())
    )
    assert set(left.index) | set(right.index) == set(dev.index)


def test_fit_shapes_shrinkage_and_laplace(swings):
    weak = ContactModel(0.1, 0.1).fit(swings)
    strong = ContactModel(100, 100).fit(swings)
    assert np.linalg.norm(strong.theta[16:]) < np.linalg.norm(weak.theta[16:])
    p = weak.predict(swings)
    assert p.shape == (len(swings),) and np.isfinite(p).all()
    assert ((p > 0) & (p < 1)).all()
    covariance = weak.laplace_covariance(swings)
    assert np.allclose(covariance, covariance.T)
    assert np.linalg.eigvalsh(covariance).min() > 0
    assert np.allclose(weak.hessian @ covariance, np.eye(weak.n_parameters), atol=1e-8)
    assert np.array_equal(weak.draw_coefficients(20, 1), weak.draw_coefficients(20, 1))
    assert weak.draw_coefficients(20, 1).shape == (weak.n_parameters, 20)
    centered = weak.player_effects("batter_id").groupby("side").effect.mean()
    assert np.allclose(centered, 0, atol=1e-12)


def test_player_quality_neutralization_and_unknown_id(swings):
    model = ContactModel().fit(swings)
    model.theta[:] = 0
    block = model.blocks["pitcher_id"]
    model.theta[block.start] = -2
    model.theta[block.start + 1] = 2
    frame = pd.DataFrame(
        {
            "batter_id": [1, 1],
            "bat_side": "L",
            "pitcher_id": [1, 2],
            "throw_side": "L",
            "is_in_zone": 1,
            "is_fb": 1,
        }
    )
    assert model.predict(frame)[0] < model.predict(frame)[1]
    assert np.allclose(model.predict(frame, neutral_pitcher=True), 0.5)
    with pytest.raises(ValueError, match="Unknown pitcher_id"):
        model.predict(frame.assign(pitcher_id=999))
    with pytest.raises(ValueError, match="Handedness"):
        model.predict(frame.assign(throw_side="R"))
    context_only = ContactModel(include_batter=False, include_pitcher=False).fit(swings)
    assert context_only.n_parameters == 16


def test_projection_averages_probabilities_with_joint_draws(swings):
    model = ContactModel().fit(swings)
    model.laplace_covariance(swings)
    draws = model.draw_coefficients(200, 2)
    reference = reference_contexts(swings)
    assert np.allclose(reference.groupby("bat_side").weight.sum(), 1)
    projected = batter_projection(model, 1, reference, draws)
    frame = reference.loc[reference.bat_side.eq("L")].assign(batter_id=1, is_in_zone=1)
    expected = frame.weight @ model.predict(frame, neutral_pitcher=True)
    assert projected.iloc[0].contact_probability == pytest.approx(expected)
    assert len(projected) == 3
    assert projected.lower_95.between(0, 1).all()
    assert projected.upper_95.between(0, 1).all()
    assert (projected.lower_95 < projected.upper_95).all()
    with pytest.raises(ValueError, match="Unknown batter ID"):
        batter_projection(model, 999, reference, draws)


def test_swings_only_and_malformed_data(swings):
    takes = swings.iloc[:10].assign(is_swing=0, is_miss=0)
    pitches = pd.concat([swings, takes], ignore_index=True)
    assert len(prepare_swings(pitches)) == len(swings)
    with pytest.raises(ValueError, match="miss must be a swing"):
        prepare_swings(pitches.assign(is_swing=0, is_miss=1))
    with pytest.raises(ValueError, match="swings only"):
        ContactModel().fit(swings.assign(is_swing=0))
    with pytest.raises(ValueError, match="positive"):
        ContactModel(batter_precision=0).fit(swings)
    with pytest.raises(ValueError, match="missing"):
        prepare_swings(swings.assign(is_fb=np.nan))


def test_validation_selection_prefers_stronger_shrinkage_when_scores_are_close():
    from contact_model import select_shrinkage

    scores = pd.DataFrame(
        {
            "batter_precision": [1, 1, 100],
            "pitcher_precision": [0.01, 1, 100],
            "log_loss": [0.4400, 0.4403, 0.4600],
        }
    )
    chosen = select_shrinkage(scores, tolerance=0.001)
    assert chosen.pitcher_precision == 1
    exact_best = select_shrinkage(scores, tolerance=0)
    assert exact_best.pitcher_precision == 0.01
    with pytest.raises(ValueError, match="nonnegative"):
        select_shrinkage(scores, tolerance=-1)


def test_reference_mix_conditions_on_batting_side_and_excludes_out_of_zone(swings):
    from scipy.special import logit

    from contact_model import CONTEXT_LEVELS

    # LHH: 75% LHP; RHH: 25% LHP, with different FB mixes within LHP.
    contexts = [
        ("L", "L", 0),
        ("L", "L", 0),
        ("L", "L", 1),
        ("L", "R", 1),
        ("R", "L", 0),
        ("R", "R", 0),
        ("R", "R", 0),
        ("R", "R", 1),
    ]
    in_zone = pd.DataFrame(contexts, columns=["bat_side", "throw_side", "is_fb"])
    in_zone["is_in_zone"] = 1
    out_of_zone = in_zone.iloc[[0]].assign(is_in_zone=0)
    reference = reference_contexts(pd.concat([in_zone, out_of_zone], ignore_index=True))
    assert reference.swings.sum() == len(in_zone)
    assert np.allclose(reference.groupby("bat_side").weight.sum(), 1)
    lhp_weights = (
        reference.loc[reference.throw_side.eq("L")].groupby("bat_side").weight.sum()
    )
    assert lhp_weights["L"] == pytest.approx(0.75)
    assert lhp_weights["R"] == pytest.approx(0.25)
    assert reference.loc[reference.swings.eq(0), "weight"].eq(0).all()

    model = ContactModel().fit(swings)
    model.theta[:] = 0
    for i, (_, throw, _, fb) in enumerate(CONTEXT_LEVELS):
        model.theta[i] = logit((0.1 if throw == "L" else 0.7) + 0.2 * fb)
    draws = model.theta[:, None]
    left = batter_projection(model, 1, reference, draws)
    right = batter_projection(model, 7, reference, draws)
    assert left.iloc[0].contact_probability == pytest.approx(0.35)
    assert right.iloc[0].contact_probability == pytest.approx(0.6)
    assert left.iloc[1].contact_probability == pytest.approx((0.1 + 0.1 + 0.3) / 3)
    assert right.iloc[1].contact_probability == pytest.approx(0.1)
    assert left.iloc[2].contact_probability == pytest.approx(0.3)
    assert right.iloc[2].contact_probability == pytest.approx(0.3)
    with pytest.raises(ValueError, match="batting side R"):
        batter_projection(model, 7, reference.loc[reference.bat_side.eq("L")], draws)
    with pytest.raises(ValueError, match="in-zone swings"):
        reference_contexts(out_of_zone)
