from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.datasets import fetch_openml
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_tweedie_deviance,
    auc,
)


def load_mtpl2(n_samples=None):
    """Fetch the French Motor Third-Party Liability Claims dataset.

    Parameters
    ----------
    n_samples: int, default=None
      number of samples to select (for faster run time). Full dataset has
      678013 samples.
    """
    # freMTPL2freq dataset from https://www.openml.org/d/41214
    df_freq = fetch_openml(data_id=41214, as_frame=True).data
    df_freq["IDpol"] = df_freq["IDpol"].astype(int)
    df_freq.set_index("IDpol", inplace=True)

    # freMTPL2sev dataset from https://www.openml.org/d/41215
    df_sev = fetch_openml(data_id=41215, as_frame=True).data

    # sum ClaimAmount over identical IDs
    df_sev = df_sev.groupby("IDpol").sum()

    df = df_freq.join(df_sev, how="left")
    df["ClaimAmount"] = df["ClaimAmount"].fillna(0)

    # unquote string fields
    for column_name in df.columns[[t is object for t in df.dtypes.values]]:
        df[column_name] = df[column_name].str.strip("'")
    return df.iloc[:n_samples]


def plot_obs_pred(
    df,
    feature,
    weight,
    observed,
    predicted,
    y_label=None,
    title=None,
    ax=None,
    fill_legend=False,
):
    """Plot observed and predicted - aggregated per feature level.

    Parameters
    ----------
    df : DataFrame
        input data
    feature: str
        a column name of df for the feature to be plotted
    weight : str
        column name of df with the values of weights or exposure
    observed : str
        a column name of df with the observed target
    predicted : DataFrame
        a dataframe, with the same index as df, with the predicted target
    fill_legend : bool, default=False
        whether to show fill_between legend
    """
    # aggregate observed and predicted variables by feature level
    df_ = df.loc[:, [feature, weight]].copy()
    df_["observed"] = df[observed] * df[weight]
    df_["predicted"] = predicted * df[weight]
    df_ = (
        df_.groupby([feature])[[weight, "observed", "predicted"]]
        .sum()
        .assign(observed=lambda x: x["observed"] / x[weight])
        .assign(predicted=lambda x: x["predicted"] / x[weight])
    )

    ax = df_.loc[:, ["observed", "predicted"]].plot(style=".", ax=ax)
    y_max = df_.loc[:, ["observed", "predicted"]].values.max() * 0.8
    p2 = ax.fill_between(
        df_.index,
        0,
        y_max * df_[weight] / df_[weight].values.max(),
        color="g",
        alpha=0.1,
    )
    if fill_legend:
        ax.legend([p2], ["{} distribution".format(feature)])
    ax.set(
        ylabel=y_label if y_label is not None else None,
        title=title if title is not None else "Train: Observed vs Predicted",
    )


def score_estimator(
    estimator,
    X_train,
    X_test,
    df_train,
    df_test,
    target,
    weights,
    tweedie_powers=None,
):
    """Evaluate an estimator on train and test sets with different metrics"""

    metrics = [
        ("D² explained", None),  # Use default scorer if it exists
        ("mean abs. error", mean_absolute_error),
        ("mean squared error", mean_squared_error),
    ]
    if tweedie_powers:
        metrics += [
            (
                "mean Tweedie dev p={:.4f}".format(power),
                partial(mean_tweedie_deviance, power=power),
            )
            for power in tweedie_powers
        ]

    res = []
    for subset_label, X, df in [
        ("train", X_train, df_train),
        ("test", X_test, df_test),
    ]:
        y, _weights = df[target], df[weights]
        for score_label, metric in metrics:
            if isinstance(estimator, tuple) and len(estimator) == 2:
                # Score the model consisting of the product of frequency and
                # severity models.
                est_freq, est_sev = estimator
                y_pred = est_freq.predict(X) * est_sev.predict(X)
            else:
                y_pred = estimator.predict(X)

            if metric is None:
                if not hasattr(estimator, "score"):
                    continue
                score = estimator.score(X, y, sample_weight=_weights)
            else:
                score = metric(y, y_pred, sample_weight=_weights)

            res.append({"subset": subset_label, "metric": score_label, "score": score})

    res = (
        pd.DataFrame(res)
        .set_index(["metric", "subset"])
        .score.unstack(-1)
        .round(4)
        .loc[:, ["train", "test"]]
    )
    return res


def lorenz_curve(y_true, y_pred, exposure):
    """
    Compute the Lorenz curve for insurance risk models.

    The Lorenz curve is a cumulative distribution plot that compares the cumulative
    share of exposure to the cumulative share of claim amounts, after sorting all
    policyholders by their predicted risk. It is commonly used to evaluate the
    discriminatory power of predictive models in insurance (e.g., for pure premium
    modeling), and forms the basis for computing the Gini index.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Observed target values, typically the actual Pure Premium per policy.

    y_pred : array-like of shape (n_samples,)
        Predicted target values produced by a model. These predictions determine
        the ranking of policies from low to high risk.

    exposure : array-like of shape (n_samples,)
        Exposure weights for each policy. These are used to compute the cumulative
        proportion of exposure and the weighted cumulative claim amounts.

    Returns
    -------
    cumulative_exposure : ndarray of shape (n_samples,)
        The cumulative proportion of exposure after sorting by predicted risk.

    cumulative_claim_amount : ndarray of shape (n_samples,)
        The cumulative proportion of (exposure-weighted) observed claim amounts.

    Notes
    -----
    - Policies are sorted from lowest to highest predicted risk.
    - The Lorenz curve starts at (0, 0) and always ends at (1, 1).
    - This function does not compute the Gini index, but its output can be passed
      to `sklearn.metrics.auc` to obtain it.
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    exposure = np.asarray(exposure)

    # order samples by increasing predicted risk:
    ranking = np.argsort(y_pred)
    ranked_exposure = exposure[ranking]
    ranked_pure_premium = y_true[ranking]
    cumulative_claim_amount = np.cumsum(ranked_pure_premium * ranked_exposure)
    cumulative_claim_amount /= cumulative_claim_amount[-1]
    cumulative_exposure = np.cumsum(ranked_exposure)
    cumulative_exposure /= cumulative_exposure[-1]
    return cumulative_exposure, cumulative_claim_amount


def plot_lorenz_curve(glm_freq, glm_sev, X_test, df_test, ax ):
    """
    Plot Lorenz curves for the frequency–severity model and baselines.

    This function computes and visualizes the Lorenz curve for the combined
    frequency × severity model by multiplying predictions from a Poisson
    (frequency) GLM and a Gamma (severity) GLM. It also plots:
    - an oracle model (perfect predictions), and
    - a random baseline (diagonal line).

    The Lorenz curve shows how well the model discriminates between high-risk
    and low-risk policies by ranking them by predicted Pure Premium and comparing
    cumulative exposure with cumulative claim amounts. The Gini index for each
    curve is computed and added to the plot legend.

    Parameters
    ----------
    glm_freq : estimator object
        Fitted model that predicts frequency (e.g., PoissonRegressor).

    glm_sev : estimator object
        Fitted model that predicts severity (e.g., GammaRegressor).

    X_test : pandas.DataFrame or array-like
        Feature matrix used for predictions.

    df_test : pandas.DataFrame
        Test dataset containing:
        - "PurePremium" : actual target values,
        - "Exposure"    : exposure weights.

    ax : matplotlib Axes
        Axes object on which the Lorenz curves will be plotted.

    Returns
    -------
    None
        The function modifies the provided Axes object in place.

    Notes
    -----
    - The Gini index is computed as: `1 - 2 * auc(cum_exposure, cum_claims)`.
    - Sorting is always done from lowest to highest predicted risk.
    - The oracle curve represents the theoretical best possible ordering.
    """
    y_pred_product = glm_freq.predict(X_test) * glm_sev.predict(X_test)

    for label, y_pred in [
        ("Frequency * Severity model", y_pred_product),
    ]:
        cum_exposure, cum_claims = lorenz_curve(
            df_test["PurePremium"], y_pred, df_test["Exposure"]
        )
        gini = 1 - 2 * auc(cum_exposure, cum_claims)
        label += " (Gini index: {:.3f})".format(gini)
        ax.plot(cum_exposure, cum_claims, linestyle="-", label=label)

    # Oracle model: y_pred == y_test
    cum_exposure, cum_claims = lorenz_curve(
        df_test["PurePremium"], df_test["PurePremium"], df_test["Exposure"]
    )
    gini = 1 - 2 * auc(cum_exposure, cum_claims)
    label = "Oracle (Gini index: {:.3f})".format(gini)
    ax.plot(cum_exposure, cum_claims, linestyle="-.", color="gray", label=label)

    # Random baseline
    ax.plot([0, 1], [0, 1], linestyle="--", color="black", label="Random baseline")
    ax.set(
        title="Lorenz Curves",
        xlabel=(
            "Cumulative proportion of exposure\n(ordered by model from safest to riskiest)"
        ),
        ylabel="Cumulative proportion of claim amounts",
    )
    ax.legend(loc="upper left")
