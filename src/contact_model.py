"""Small crossed-effects contact model with fixed Gaussian shrinkage scales.

The likelihood contains swings only. Context is batting side, throwing side,
zone, and fastball status. Player effects are jointly estimated; there are no
player-specific context slopes. Laplace intervals condition on chosen penalties.
"""

from dataclasses import dataclass
from itertools import product
from time import perf_counter

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import sparse
from scipy.linalg import cho_factor, cho_solve, cholesky
from scipy.optimize import minimize
from scipy.special import expit, logit
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split

CONTEXT = ["bat_side", "throw_side", "is_in_zone", "is_fb"]
CONTEXT_LEVELS = list(product(["L", "R"], ["L", "R"], [0, 1], [0, 1]))


def prepare_swings(pitches: pd.DataFrame) -> pd.DataFrame:
    """Validate the pitch schema and retain original row IDs for swings."""
    required = ["batter_id", "pitcher_id", *CONTEXT, "is_swing", "is_miss"]
    if not pitches.index.is_unique:
        raise ValueError("Pitch rows must have unique index values.")
    if pitches[required].isna().any().any():
        raise ValueError("Review missing values before modeling.")
    if not pitches[["bat_side", "throw_side"]].isin(["L", "R"]).all().all():
        raise ValueError("Handedness must be L or R.")
    binary = ["is_in_zone", "is_fb", "is_swing", "is_miss"]
    if not pitches[binary].isin([0, 1]).all().all():
        raise ValueError("Pitch indicators must be binary.")
    if (pitches.is_miss > pitches.is_swing).any():
        raise ValueError("A miss must be a swing.")
    for player, hand in [("batter_id", "bat_side"), ("pitcher_id", "throw_side")]:
        if pitches.groupby(player)[hand].nunique().gt(1).any():
            raise ValueError(f"This first-pass model assumes one side per {player}.")
    swings = pitches.loc[pitches.is_swing.eq(1)].copy()
    swings["contact"] = 1 - swings.is_miss
    return swings


def split_swings(
    swings: pd.DataFrame, seed: int = 42, max_attempts: int = 100
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, int]:
    """Split 60/20/20 by context, requiring individual coverage, without targets.

    If coverage fails, use the next seed based solely on IDs/context. Never
    select a seed using contact outcomes. Raise rather than silently dropping
    a sparse player or weakening the requirement.
    """
    strata = swings[CONTEXT].astype(str).agg("_".join, axis=1)
    players = {key: set(swings[key]) for key in ["batter_id", "pitcher_id"]}
    for split_seed in range(seed, seed + max_attempts):
        try:
            development, test = train_test_split(
                swings.index, test_size=0.2, random_state=split_seed, stratify=strata
            )
            train, validation = train_test_split(
                development,
                test_size=0.25,
                random_state=split_seed,
                stratify=strata.loc[development],
            )
        except ValueError as exc:
            raise ValueError(
                "Too few swings in one or more contexts to stratify."
            ) from exc
        parts = tuple(swings.loc[index].copy() for index in [train, validation, test])
        if all(set(part[key]) == players[key] for part in parts for key in players):
            return (*parts, split_seed)
    raise ValueError("Cannot achieve individual coverage. Review sparse players.")


def pair_sensitivity_split(
    development: pd.DataFrame, seed: int = 43
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Make one pair-grouped sensitivity split using development data only."""
    groups = pd.MultiIndex.from_frame(development[["batter_id", "pitcher_id"]])
    codes, _ = pd.factorize(groups)
    left, right = next(
        GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed).split(
            development, groups=codes
        )
    )
    return development.iloc[left].copy(), development.iloc[right].copy()


def context_indices(frame: pd.DataFrame) -> NDArray[np.int64]:
    """Encode the predeclared 16 contexts without learning from held-out data."""
    if not frame[["bat_side", "throw_side"]].isin(["L", "R"]).all().all():
        raise ValueError("Handedness must be L or R.")
    if not frame[["is_in_zone", "is_fb"]].isin([0, 1]).all().all():
        raise ValueError("Zone and fastball indicators must be 0 or 1.")
    return (
        frame.bat_side.eq("R").to_numpy(dtype=int) * 8
        + frame.throw_side.eq("R").to_numpy(dtype=int) * 4
        + frame.is_in_zone.to_numpy(dtype=int) * 2
        + frame.is_fb.to_numpy(dtype=int)
    )


def objective_gradient(
    theta: NDArray, design: sparse.csr_matrix, y: NDArray, precision: NDArray
) -> tuple[float, NDArray]:
    """Stable summed Bernoulli negative log posterior and analytic gradient."""
    eta = np.asarray(design @ theta).ravel()
    loss = np.sum(np.logaddexp(0, eta) - y * eta) + 0.5 * np.dot(
        precision * theta, theta
    )
    gradient = np.asarray(design.T @ (expit(eta) - y)).ravel() + precision * theta
    return float(loss), gradient


@dataclass
class ContactModel:
    """Penalized logistic model; precisions are fixed, not learned variances."""

    batter_precision: float = 10.0
    pitcher_precision: float = 10.0
    include_batter: bool = True
    include_pitcher: bool = True
    context_sd: float = 2.5

    def fit(self, swings: pd.DataFrame) -> "ContactModel":
        """Fit a MAP estimate and check convergence. No environment changes."""
        if len(swings) == 0 or not swings.contact.isin([0, 1]).all():
            raise ValueError("Training data must contain binary contact outcomes.")
        if not swings.is_swing.eq(1).all():
            raise ValueError("The contact likelihood accepts swings only.")
        scales = [self.batter_precision, self.pitcher_precision, self.context_sd]
        if not np.isfinite(scales).all() or min(scales) <= 0:
            raise ValueError("Prior scales and precisions must be positive and finite.")
        self.player_ids = {}
        self.player_sides = {}
        self.blocks = {}
        self.precision = np.full(16, 1 / self.context_sd**2)
        offset = 16
        for key, hand, enabled, precision in [
            ("batter_id", "bat_side", self.include_batter, self.batter_precision),
            ("pitcher_id", "throw_side", self.include_pitcher, self.pitcher_precision),
        ]:
            if enabled:
                sides = swings.groupby(key)[hand].agg(["first", "nunique"])
                if sides["nunique"].gt(1).any():
                    raise ValueError(f"One side per {key} is required.")
                ids = sides.index.to_numpy()
                self.player_ids[key] = ids
                self.player_sides[key] = sides["first"].to_numpy()
                self.blocks[key] = slice(offset, offset + len(ids))
                offset += len(ids)
                self.precision = np.r_[self.precision, np.full(len(ids), precision)]
        self.n_parameters = offset
        design = self.design_matrix(swings)
        y = swings.contact.to_numpy(dtype=float)
        initial = np.zeros(offset)
        initial[:16] = logit(np.clip(y.mean(), 0.01, 0.99))
        start = perf_counter()
        result = minimize(
            objective_gradient,
            initial,
            args=(design, y, self.precision),
            method="L-BFGS-B",
            jac=True,
            options={"maxiter": 2000, "ftol": 1e-13, "gtol": 1e-6},
        )
        self.gradient_max = float(np.max(np.abs(result.jac)))
        if (
            not result.success
            or self.gradient_max > 0.01
            or not np.isfinite(result.x).all()
        ):
            raise RuntimeError(
                f"Fit failed: {result.message}; max gradient={self.gradient_max:.3g}"
            )
        self.theta = result.x
        self.fit_seconds = perf_counter() - start
        self.iterations = result.nit
        self.training_counts = {
            key: swings.groupby(key).size() for key in ["batter_id", "pitcher_id"]
        }
        self.covariance = None
        self.hessian = None
        return self

    def design_matrix(
        self,
        frame: pd.DataFrame,
        neutral_batter: bool = False,
        neutral_pitcher: bool = False,
    ) -> sparse.csr_matrix:
        """Encode actual or neutral players using training IDs only.

        Neutral means mean fitted logit effect among training players of the
        specified side (equal player weights). Equivalently, recenter effects
        within handedness and set the neutral deviation to zero. Unknown IDs
        raise; they must not silently receive zero uncertainty.
        """
        n = len(frame)
        rows = list(range(n))
        cols = context_indices(frame).tolist()
        values = [1.0] * n
        for key, hand, neutral in [
            ("batter_id", "bat_side", neutral_batter),
            ("pitcher_id", "throw_side", neutral_pitcher),
        ]:
            if key not in self.blocks:
                continue
            offset = self.blocks[key].start
            ids = self.player_ids[key]
            sides = self.player_sides[key]
            if neutral:
                for side in ["L", "R"]:
                    targets = np.flatnonzero(frame[hand].eq(side).to_numpy())
                    members = np.flatnonzero(sides == side)
                    if len(targets) and not len(members):
                        raise ValueError(f"No training {key} players for side {side}.")
                    for row in targets:
                        rows.extend([int(row)] * len(members))
                        cols.extend((offset + members).tolist())
                        values.extend([1 / len(members)] * len(members))
            else:
                if key not in frame:
                    raise ValueError(f"Supply {key}, or request a neutral player.")
                positions = pd.Index(ids).get_indexer(frame[key])
                if (positions < 0).any():
                    unknown = frame.loc[positions < 0, key].unique().tolist()
                    raise ValueError(
                        f"Unknown {key}: {unknown}. Use an explicit neutral player."
                    )
                if not np.array_equal(sides[positions], frame[hand].to_numpy()):
                    raise ValueError(f"Handedness does not match training {key}.")
                rows.extend(range(n))
                cols.extend((offset + positions).tolist())
                values.extend([1.0] * n)
        return sparse.csr_matrix((values, (rows, cols)), shape=(n, self.n_parameters))

    def predict(
        self,
        frame: pd.DataFrame,
        neutral_batter: bool = False,
        neutral_pitcher: bool = False,
    ) -> NDArray:
        """Return probabilities including all requested player/context effects."""
        return expit(
            self.design_matrix(frame, neutral_batter, neutral_pitcher) @ self.theta
        )

    def laplace_covariance(self, training: pd.DataFrame) -> NDArray:
        """Invert posterior curvature conditional on fixed prior precisions."""
        design = self.design_matrix(training)
        p = expit(design @ self.theta)
        self.hessian = (
            design.T @ design.multiply((p * (1 - p))[:, None])
        ).toarray() + np.diag(self.precision)
        factor = cho_factor(self.hessian, lower=True)
        self.covariance = cho_solve(factor, np.eye(self.n_parameters))
        self.hessian_condition = float(np.linalg.cond(self.hessian))
        return self.covariance

    def draw_coefficients(self, n_draws: int = 2000, seed: int = 44) -> NDArray:
        """Draw joint approximate posterior coefficients, including covariance."""
        if self.covariance is None:
            raise ValueError("Compute Laplace covariance before drawing coefficients.")
        rng = np.random.default_rng(seed)
        return self.theta[:, None] + cholesky(self.covariance, lower=True) @ rng.normal(
            size=(self.n_parameters, n_draws)
        )

    def player_effects(self, player: str) -> pd.DataFrame:
        """Show deviations centered within handedness, and training swing counts."""
        block = self.blocks[player]
        result = pd.DataFrame(
            {"side": self.player_sides[player], "effect": self.theta[block]},
            index=pd.Index(self.player_ids[player], name=player),
        )
        result["effect"] -= result.groupby("side").effect.transform("mean")
        result["swing_count"] = self.training_counts[player]
        return result


def performance(frame: pd.DataFrame, probability: NDArray) -> dict:
    """Proper scores plus discrimination and a batter-balanced summary."""
    y = frame.contact.to_numpy()
    p = np.clip(probability, 1e-12, 1 - 1e-12)
    row_loss = -(y * np.log(p) + (1 - y) * np.log1p(-p))
    balanced = (
        pd.DataFrame(
            {
                "batter": frame.batter_id.to_numpy(),
                "log_loss": row_loss,
                "brier": (y - p) ** 2,
            }
        )
        .groupby("batter")
        .mean()
        .mean()
    )
    return {
        "swings": len(y),
        "log_loss": log_loss(y, p, labels=[0, 1]),
        "brier": brier_score_loss(y, p),
        "roc_auc": roc_auc_score(y, p) if np.unique(y).size == 2 else np.nan,
        "batter_balanced_log_loss": balanced.log_loss,
        "batter_balanced_brier": balanced.brier,
        "observed": y.mean(),
        "predicted": p.mean(),
        "observed_minus_predicted": y.mean() - p.mean(),
    }


def subgroup_performance(
    frame: pd.DataFrame, probability: NDArray, keys: list
) -> pd.DataFrame:
    """Score contexts or other predeclared subgroups, retaining sample sizes."""
    work = frame.assign(prediction=probability)
    rows = []
    for group, subset in work.groupby(keys, observed=True):
        group = group if isinstance(group, tuple) else (group,)
        rows.append(
            {
                **dict(zip(keys, group)),
                **performance(subset, subset.prediction.to_numpy()),
            }
        )
    return pd.DataFrame(rows).set_index(keys)


def reliability(
    frame: pd.DataFrame, probability: NDArray, max_bins: int = 10
) -> pd.DataFrame:
    """Quantile-bin calibration with counts; coarse bins for small subgroups."""
    bins = max(1, min(max_bins, len(frame) // 30))
    work = pd.DataFrame(
        {"observed": frame.contact.to_numpy(), "predicted": probability}
    )
    if work.predicted.nunique() == 1:
        work["bin"] = 0
    else:
        work["bin"] = pd.qcut(work.predicted, q=bins, duplicates="drop")
    return work.groupby("bin", observed=True).agg(
        predicted=("predicted", "mean"),
        observed=("observed", "mean"),
        count=("observed", "size"),
    )


def select_shrinkage(trials: pd.DataFrame, tolerance: float = 0.001) -> pd.Series:
    """Prefer stronger priors among near-equivalent validation log losses.

    Among scores within tolerance of the minimum, minimize total player-effect
    prior variance, then log loss. The tolerance is a development setting;
    neither test scores nor test outcomes enter this rule.
    """
    if tolerance < 0 or not np.isfinite(tolerance):
        raise ValueError("Validation tolerance must be finite and nonnegative.")
    columns = [
        key for key in ["batter_precision", "pitcher_precision"] if key in trials
    ]
    if (
        not columns
        or trials.empty
        or not np.isfinite(trials[columns + ["log_loss"]]).all().all()
    ):
        raise ValueError(
            "Supply finite validation scores and positive prior precisions."
        )
    if not trials[columns].gt(0).all().all():
        raise ValueError("Prior precisions must be positive.")
    eligible = trials.loc[trials.log_loss <= trials.log_loss.min() + tolerance].copy()
    eligible["total_prior_variance"] = (1 / eligible[columns]).sum(axis=1)
    selected = eligible.sort_values(["total_prior_variance", "log_loss"]).iloc[0]
    return selected.drop("total_prior_variance")


def reference_contexts(development: pd.DataFrame) -> pd.DataFrame:
    """Freeze joint throw-side/FB weights within batting side and IZ swings.

    Weights sum to one for each represented batting side. Every hitter of that
    side uses the same mix, estimated from development data only. Zero-count
    throw-side/FB combinations remain explicit in the returned table.
    """
    iz = development.loc[development.is_in_zone.eq(1)]
    batting_sides = sorted(development.bat_side.unique())
    index = pd.MultiIndex.from_product(
        [batting_sides, ["L", "R"], [0, 1]],
        names=["bat_side", "throw_side", "is_fb"],
    )
    counts = (
        iz.groupby(["bat_side", "throw_side", "is_fb"])
        .size()
        .reindex(index, fill_value=0)
    )
    totals = counts.groupby(level="bat_side").transform("sum")
    if counts.empty or totals.eq(0).any():
        raise ValueError("Reference mix requires in-zone swings for each batting side.")
    return (
        counts.rename("swings")
        .reset_index()
        .assign(weight=(counts / totals).to_numpy())
    )


def batter_projection(
    model: ContactModel, batter_id: int, reference: pd.DataFrame, draws: NDArray
) -> pd.DataFrame:
    """Three PDF answers using neutral pitching and the batter's side-specific mix."""
    ids = pd.Index(model.player_ids["batter_id"])
    position = ids.get_indexer([batter_id])[0]
    if position < 0:
        raise ValueError(f"Unknown batter ID {batter_id}; choose a training batter ID.")
    side = model.player_sides["batter_id"][position]
    side_reference = reference.loc[reference.bat_side.eq(side)]
    if side_reference.empty:
        raise ValueError(f"No reference mix for batting side {side}.")
    rows = []
    for name, subset in [
        ("In-zone pitches", side_reference),
        (
            "In-zone pitches vs LHP",
            side_reference.loc[side_reference.throw_side.eq("L")],
        ),
        (
            "In-zone fastballs vs LHP",
            side_reference.loc[
                side_reference.throw_side.eq("L") & side_reference.is_fb.eq(1)
            ],
        ),
    ]:
        frame = subset.assign(batter_id=batter_id, bat_side=side, is_in_zone=1)
        # The fully conditioned FB/LHP answer exists even if its reference weight is zero.
        weights = (
            np.ones(len(frame))
            if name.endswith("fastballs vs LHP")
            else frame.weight.to_numpy()
        )
        if weights.sum() <= 0:
            raise ValueError(f"No reference support for {name}.")
        weights = weights / weights.sum()
        design = model.design_matrix(frame, neutral_pitcher=True)
        simulated_rates = weights @ expit(design @ draws)
        rows.append(
            {
                "scenario": name,
                "contact_probability": float(weights @ expit(design @ model.theta)),
                "lower_95": float(np.quantile(simulated_rates, 0.025)),
                "upper_95": float(np.quantile(simulated_rates, 0.975)),
            }
        )
    return pd.DataFrame(rows).set_index("scenario")


def simulation_recovery(
    model: ContactModel,
    development: pd.DataFrame,
    repetitions: int = 20,
    seed: int = 46,
) -> pd.DataFrame:
    """Check Laplace recovery under the assumed hierarchy and actual design.

    Simulate player effects from the selected Gaussian scales and Bernoulli
    outcomes on the development matchup structure. Context logits are held at
    the fitted values. Evaluate the fully conditioned neutral-pitcher FB/LHP
    probability for every batter. This is a model-conditional check, not proof
    of coverage under misspecification or shrinkage-selection uncertainty.
    """
    rng = np.random.default_rng(seed)
    design = model.design_matrix(development)
    ids = model.player_ids["batter_id"]
    targets = pd.DataFrame(
        {
            "batter_id": ids,
            "bat_side": model.player_sides["batter_id"],
            "throw_side": "L",
            "is_in_zone": 1,
            "is_fb": 1,
        }
    )
    target_design = model.design_matrix(targets, neutral_pitcher=True).toarray()
    counts = model.training_counts["batter_id"].reindex(ids).to_numpy()
    records = []
    for repetition in range(repetitions):
        truth = model.theta.copy()
        truth[16:] = rng.normal(size=model.n_parameters - 16) / np.sqrt(
            model.precision[16:]
        )
        simulated = development.copy()
        simulated["contact"] = rng.binomial(1, expit(design @ truth))
        simulated["is_miss"] = 1 - simulated.contact
        fitted = ContactModel(
            model.batter_precision, model.pitcher_precision, context_sd=model.context_sd
        ).fit(simulated)
        covariance = fitted.laplace_covariance(simulated)
        predicted_logit = target_design @ fitted.theta
        true_logit = target_design @ truth
        sd = np.sqrt(np.einsum("ij,ij->i", target_design @ covariance, target_design))
        for j, batter_id in enumerate(ids):
            records.append(
                {
                    "replication": repetition,
                    "batter_id": batter_id,
                    "swing_count": counts[j],
                    "true_probability": expit(true_logit[j]),
                    "estimated_probability": expit(predicted_logit[j]),
                    "error": expit(predicted_logit[j]) - expit(true_logit[j]),
                    "covered_95": abs(predicted_logit[j] - true_logit[j])
                    <= 1.96 * sd[j],
                    "interval_width": expit(predicted_logit[j] + 1.96 * sd[j])
                    - expit(predicted_logit[j] - 1.96 * sd[j]),
                }
            )
    return pd.DataFrame(records)


def predictive_context_check(
    model: ContactModel, frame: pd.DataFrame, draws: NDArray, seed: int = 47
) -> pd.DataFrame:
    """Replicate held-out contact counts with parameter and Bernoulli uncertainty."""
    rng = np.random.default_rng(seed)
    design = model.design_matrix(frame)
    replicated = rng.binomial(1, expit(design @ draws))
    context = context_indices(frame)
    records = []
    for i, values in enumerate(CONTEXT_LEVELS):
        mask = context == i
        if not mask.any():
            continue
        fraction = replicated[mask].mean(axis=0)
        observed = frame.loc[mask, "contact"].mean()
        lower, upper = np.quantile(fraction, [0.025, 0.975])
        records.append(
            {
                **dict(zip(CONTEXT, values)),
                "swings": int(mask.sum()),
                "observed": observed,
                "replicated_mean": fraction.mean(),
                "lower_95": lower,
                "upper_95": upper,
                "within_predictive_interval": lower <= observed <= upper,
            }
        )
    return pd.DataFrame(records).set_index(CONTEXT)
