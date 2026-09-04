"""تحلیل کامل پروژه آمار و احتمال روی فایل snapshotها."""

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib_cache").resolve()))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


TIMES = [5, 10, 20, 30, 1440]
SEED = 1405
L2_PENALTY = 0.03
LOCAL_NEIGHBORS = 30
FN_COST_RATIO = 2
OUT = Path("output")
FIG = OUT / "figures"
TAB = OUT / "tables"


def save_plot(name):
    plt.tight_layout()
    plt.savefig(FIG / name, dpi=170, bbox_inches="tight")
    plt.close()


def load_and_clean(filename):
    data = pd.read_csv(filename)
    needed = {
        "channel", "message_id", "subscribers", "channel_type", "published_at",
        "checkpoint_min", "views", "reactions", "forwards",
    }
    missing = needed - set(data.columns)
    if missing:
        raise ValueError(f"این ستون ها در فایل نیستند: {sorted(missing)}")

    data["published_at"] = pd.to_datetime(data["published_at"], utc=True)
    for col in ["message_id", "subscribers", "checkpoint_min", "views", "reactions", "forwards"]:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    if "actual_age_min" in data.columns:
        data["actual_age_min"] = pd.to_numeric(data["actual_age_min"], errors="coerce")

    data = data.dropna(subset=list(needed))
    data = data[data["checkpoint_min"].isin(TIMES)]
    data = data[(data["subscribers"] > 0) & (data[["views", "reactions", "forwards"]] >= 0).all(axis=1)]
    data = data.drop_duplicates(["channel", "message_id", "checkpoint_min"], keep="last")
    data = data.sort_values(["channel", "message_id", "checkpoint_min"])

    # شمارنده ها تجمعی اند. افت کوچک معمولا مشکل ثبت یا API است.
    for col in ["views", "reactions", "forwards"]:
        data[col] = data.groupby(["channel", "message_id"])[col].cummax()

    post_quality = data.groupby(["channel", "message_id"], as_index=False).agg(
        channel_type=("channel_type", "first"),
        subscribers=("subscribers", "max"),
        checkpoint_count=("checkpoint_min", "nunique"),
    )
    post_quality["complete"] = post_quality["checkpoint_count"] == len(TIMES)
    completion_by_channel = post_quality.groupby(["channel", "channel_type"], as_index=False).agg(
        total_posts=("message_id", "size"),
        complete_posts=("complete", "sum"),
    )
    completion_by_channel["completion_rate"] = (
        completion_by_channel["complete_posts"] / completion_by_channel["total_posts"]
    )
    completion_by_channel.to_csv(TAB / "data_quality_completion_by_channel.csv", index=False)

    complete = post_quality.loc[post_quality["complete"], ["channel", "message_id"]]
    complete = pd.MultiIndex.from_frame(complete)
    before = data[["channel", "message_id"]].drop_duplicates().shape[0]
    data = data.set_index(["channel", "message_id"]).loc[complete].reset_index()
    after = data[["channel", "message_id"]].drop_duplicates().shape[0]

    quality = {
        "unique_posts_before_complete_filter": int(before),
        "complete_posts": int(after),
        "completion_rate": float(after / before),
        "min_channel_completion_rate": float(completion_by_channel["completion_rate"].min()),
        "max_channel_completion_rate": float(completion_by_channel["completion_rate"].max()),
    }
    if "actual_age_min" in data.columns:
        # تاخیر نقطه نهایی روی همه snapshotهای نهایی فایل خام حساب می شود نه فقط پست های کامل
        raw_again = pd.read_csv(filename, usecols=["checkpoint_min", "actual_age_min"])
        raw_again["checkpoint_min"] = pd.to_numeric(raw_again["checkpoint_min"], errors="coerce")
        raw_again["actual_age_min"] = pd.to_numeric(raw_again["actual_age_min"], errors="coerce")
        final_delay = raw_again.loc[raw_again["checkpoint_min"] == 1440, "actual_age_min"].dropna() - 1440
        if len(final_delay):
            quality.update({
                "final_snapshot_count": int(len(final_delay)),
                "final_delay_median_min": float(final_delay.median()),
                "final_delay_p95_min": float(final_delay.quantile(0.95)),
                "final_delay_max_min": float(final_delay.max()),
            })
    (TAB / "data_quality_summary.json").write_text(json.dumps(quality, indent=2), encoding="utf-8")

    print(f"پست کامل برای تحلیل: {after} از {before}")
    if after < 50:
        raise ValueError("کمتر از 50 پست کامل مانده؛ جمع آوری را ادامه بده.")
    return data


def add_popularity_score(data):
    data = data.copy()
    data["view_rate"] = data["views"] / data["subscribers"]
    data["reaction_rate"] = data["reactions"] / data["views"].clip(lower=1)
    data["forward_rate"] = data["forwards"] / data["views"].clip(lower=1)

    groups = ["channel", "checkpoint_min"]
    data["med_views"] = data.groupby(groups)["views"].transform("median").clip(lower=1)
    data["med_rr"] = data.groupby(groups)["reaction_rate"].transform("median")
    data["med_fr"] = data.groupby(groups)["forward_rate"].transform("median")

    view_part = np.log1p(data["views"] / data["med_views"]) / np.log(2)
    react_part = np.log1p(data["reaction_rate"] / data["med_rr"].clip(lower=1e-8)) / np.log(2)
    forward_part = np.log1p(data["forward_rate"] / data["med_fr"].clip(lower=1e-8)) / np.log(2)

    # اگر واکنش یا فوروارد در یک کانال بسته باشد، وزن آن را صفر می کنیم.
    wr = np.where(data["med_rr"] > 0, 0.30, 0.0)
    wf = np.where(data["med_fr"] > 0, 0.20, 0.0)
    wv = np.full(len(data), 0.50)
    total_w = wv + wr + wf
    data["score"] = 100 * (wv * view_part + wr * react_part + wf * forward_part) / total_w
    return data


def make_wide(data):
    base_cols = ["channel", "message_id", "subscribers", "channel_type", "published_at"]
    base = data[base_cols].drop_duplicates(["channel", "message_id"]).set_index(["channel", "message_id"])

    for metric in ["score", "views", "reactions", "forwards", "reaction_rate", "forward_rate"]:
        p = data.pivot(index=["channel", "message_id"], columns="checkpoint_min", values=metric)
        if metric == "score":
            p.columns = [f"s{int(t)}" for t in p.columns]
        else:
            p.columns = [f"{metric}_{int(t)}" for t in p.columns]
        base = base.join(p)

    base = base.reset_index()
    base["publish_hour"] = base["published_at"].dt.hour
    return base


def part1_descriptive(posts):
    score = posts["s1440"].to_numpy()
    mean = score.mean()
    sd = score.std(ddof=1)
    q1, median, q3 = np.quantile(score, [0.25, 0.5, 0.75])
    positive = score[score > 0]

    desc = {
        "count": len(score),
        "arithmetic_mean": mean,
        "subscriber_weighted_mean": np.average(score, weights=posts["subscribers"]),
        "geometric_mean": stats.gmean(positive),
        "harmonic_mean": stats.hmean(positive),
        "median": median,
        "q1": q1,
        "q3": q3,
        "range": score.max() - score.min(),
        "iqr": q3 - q1,
        "sample_variance": score.var(ddof=1),
        "sample_std": sd,
        "cv": sd / mean,
        "skewness": stats.skew(score, bias=False),
        "excess_kurtosis": stats.kurtosis(score, fisher=True, bias=False),
    }
    pd.Series(desc, name="value").to_csv(TAB / "part1_descriptive.csv")

    cheb = []
    for k in [2, 3]:
        actual = np.mean(np.abs(score - mean) <= k * sd)
        cheb.append({"k": k, "theoretical_min": 1 - 1 / k**2, "actual": actual})
    pd.DataFrame(cheb).to_csv(TAB / "part1_chebyshev.csv", index=False)

    counts, edges = np.histogram(score, bins=8)
    freq = pd.DataFrame({
        "lower": edges[:-1], "upper": edges[1:], "frequency": counts,
        "relative_frequency": counts / len(score), "cumulative_frequency": np.cumsum(counts),
    })
    freq.to_csv(TAB / "part1_frequency_table.csv", index=False)

    x = np.linspace(score.min(), score.max(), 300)
    plt.figure(figsize=(7, 4.5))
    plt.hist(score, bins=16, density=True, alpha=0.65, color="#4d7ea8", edgecolor="white")
    plt.plot(x, stats.norm.pdf(x, mean, sd), color="#d1495b", lw=2, label="Fitted normal")
    plt.xlabel("Popularity score at 24h")
    plt.ylabel("Density")
    plt.title("Popularity score and fitted normal distribution")
    plt.legend()
    save_plot("part1_histogram_normal.png")

    plt.figure(figsize=(5.5, 5.5))
    stats.probplot(score, dist="norm", plot=plt)
    plt.title("Normal Q-Q plot of popularity score")
    save_plot("part1_qq_plot.png")

    plt.figure(figsize=(6.5, 4.2))
    plt.step(edges[1:], np.cumsum(counts), where="post", color="#3a7d44")
    plt.scatter(edges[1:], np.cumsum(counts), color="#3a7d44", s=22)
    plt.xlabel("Popularity score (upper class boundary)")
    plt.ylabel("Cumulative frequency")
    plt.title("Ogive")
    plt.grid(alpha=0.25)
    save_plot("part1_ogive.png")

    channel_names = sorted(posts["channel"].unique())
    plt.figure(figsize=(10, 5))
    plt.boxplot([posts.loc[posts["channel"] == c, "s1440"] for c in channel_names], tick_labels=channel_names)
    plt.xticks(rotation=35, ha="right")
    plt.ylabel("Popularity score at 24h")
    plt.title("Score by channel")
    save_plot("part1_boxplot_channels.png")

    # برای نمودار ساقه و برگ 80 مشاهده یکنواخت برداشته شده تا خروجی شلوغ نشود.
    sample = np.sort(score)[np.linspace(0, len(score) - 1, min(80, len(score))).astype(int)]
    stems = {}
    for value in np.rint(sample).astype(int):
        stems.setdefault(value // 10, []).append(abs(value) % 10)
    stem_text = "\n".join(f"{stem:>3} | {' '.join(map(str, leaves))}" for stem, leaves in stems.items())
    (TAB / "part1_stem_leaf.txt").write_text(stem_text, encoding="utf-8")

    top_cols = ["channel", "message_id", "subscribers", "s1440", "views_1440", "reactions_1440", "forwards_1440"]
    posts.nlargest(10, "s1440")[top_cols].to_csv(TAB / "part1_top10.csv", index=False)
    return desc


def part2_early_prediction(posts):
    result = []
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    y = posts["s1440"].to_numpy()

    for ax, t in zip(axes.flat, [5, 10, 20, 30]):
        x = posts[f"s{t}"].to_numpy()
        covariance = np.cov(x, y, ddof=1)[0, 1]
        r, p_value = stats.pearsonr(x, y)
        result.append({
            "time_min": t, "covariance": covariance, "pearson_r": r,
            "p_value": p_value, "significant_0.05": p_value < 0.05,
            "substantial_abs_r_ge_0.6": abs(r) >= 0.6,
        })
        slope, intercept = np.polyfit(x, y, 1)
        line_x = np.linspace(x.min(), x.max(), 100)
        ax.scatter(x, y, s=13, alpha=0.45, color="#496f8a")
        ax.plot(line_x, slope * line_x + intercept, color="#c3423f", lw=1.8)
        ax.set_title(f"{t} min: r={r:.3f}, p={p_value:.3g}")
        ax.set_xlabel(f"Score at {t} min")
        ax.set_ylabel("Score at 24h")
        ax.grid(alpha=0.18)

    fig.suptitle("Early popularity versus final popularity", y=1.01)
    save_plot("part2_scatter_regression.png")
    table = pd.DataFrame(result)
    table.to_csv(TAB / "part2_correlations.csv", index=False)
    return table


def part3_stability(posts):
    q20 = posts["s20"].quantile(0.80)
    q24 = posts["s1440"].quantile(0.80)
    early = posts["s20"] >= q20
    final = posts["s1440"] >= q24
    observed = pd.crosstab(early, final).reindex(index=[False, True], columns=[False, True], fill_value=0)
    chi2, chi_p, dof, expected = stats.chi2_contingency(observed, correction=False)

    if (expected < 5).any():
        test_name = "Fisher exact"
        test_stat, p_value = stats.fisher_exact(observed.to_numpy())
    else:
        test_name = "Chi-square"
        test_stat, p_value = chi2, chi_p

    cramer_v = np.sqrt(chi2 / observed.to_numpy().sum())
    stayed = final[early].mean()
    became = final[~early].mean()
    answer = {
        "q80_at_20min": q20,
        "q80_at_24h": q24,
        "percent_early_popular_still_popular": 100 * stayed,
        "percent_early_not_popular_became_popular": 100 * became,
        "test": test_name,
        "test_statistic": test_stat,
        "degrees_of_freedom": int(dof),
        "p_value": p_value,
        "alpha": 0.05,
        "reject_independence": bool(p_value < 0.05),
        "cramers_v": cramer_v,
        "all_expected_at_least_5": bool((expected >= 5).all()),
    }
    observed.to_csv(TAB / "part3_observed_table.csv")
    pd.DataFrame(expected, index=observed.index, columns=observed.columns).to_csv(TAB / "part3_expected_table.csv")
    (TAB / "part3_hypothesis_result.json").write_text(json.dumps(answer, indent=2), encoding="utf-8")
    return early, final, answer


def safe_mean(series):
    return float(series.mean()) if len(series) else np.nan


def part4_bayes(posts, final_popular):
    events = pd.DataFrame(index=posts.index)
    events["E1_reaction_top30_at_10"] = posts["reaction_rate_10"] >= posts["reaction_rate_10"].quantile(0.70)
    events["E2_score_top20_at_20"] = posts["s20"] >= posts["s20"].quantile(0.80)
    events["E3_score_top20_at_30"] = posts["s30"] >= posts["s30"].quantile(0.80)
    prior = float(final_popular.mean())

    rows = []
    for name in events.columns:
        event = events[name]
        empirical = safe_mean(final_popular[event])
        like_pop = safe_mean(event[final_popular])
        like_not = safe_mean(event[~final_popular])
        total_event = like_pop * prior + like_not * (1 - prior)
        bayes = like_pop * prior / total_event
        rows.append({
            "event": name, "prior": prior, "P_event_given_popular": like_pop,
            "P_event_given_not_popular": like_not, "P_event_total": total_event,
            "empirical_P_popular_given_event": empirical, "bayes_result": bayes,
            "absolute_difference": abs(empirical - bayes),
        })
    pd.DataFrame(rows).to_csv(TAB / "part4_bayes_events.csv", index=False)

    def path_values(wanted):
        mask = pd.Series(True, index=posts.index)
        current = prior
        values = [current]
        for name in events.columns:
            event_now = events[name] == wanted
            like_pop = safe_mean(event_now[mask & final_popular])
            like_not = safe_mean(event_now[mask & ~final_popular])
            denominator = like_pop * current + like_not * (1 - current)
            current = like_pop * current / denominator if denominator else current
            mask &= event_now
            values.append(current)
        return values

    good = path_values(True)
    bad = path_values(False)
    labels = ["Prior", "10 min (E1)", "20 min (E2)", "30 min (E3)"]
    paths = pd.DataFrame({"stage": labels, "good_path": good, "bad_path": bad})
    paths.to_csv(TAB / "part4_sequential_paths.csv", index=False)

    plt.figure(figsize=(7.5, 4.6))
    plt.plot(labels, good, marker="o", lw=2, label="All events observed")
    plt.plot(labels, bad, marker="s", lw=2, ls="--", label="No events observed")
    plt.axhline(prior, color="gray", ls=":", label=f"Prior = {prior:.1%}")
    plt.ylim(0, 1)
    plt.ylabel("P(popular at 24h)")
    plt.title("Sequential Bayesian update")
    plt.grid(alpha=0.2)
    plt.legend()
    save_plot("part4_bayesian_update.png")
    return paths


def random_split(n, test_fraction=0.2):
    rng = np.random.default_rng(SEED)
    order = rng.permutation(n)
    test_size = max(1, int(round(n * test_fraction)))
    return order[test_size:], order[:test_size]


def sigmoid(z):
    z = np.clip(z, -35, 35)
    return 1 / (1 + np.exp(-z))


def fit_logistic_by_hand(x, y):
    x1 = np.column_stack([np.ones(len(x)), x])
    beta = np.zeros(x1.shape[1])
    # جریمه کوچک L2 فقط برای ضرایب ویژگی ها است و عرض از مبدا جریمه نمی شود
    penalty = np.diag([0] + [L2_PENALTY] * x.shape[1])
    for _ in range(60):
        p = sigmoid(x1 @ beta)
        w = p * (1 - p) + 1e-7
        hessian = x1.T @ (w[:, None] * x1) + penalty
        gradient = x1.T @ (y - p) - penalty @ beta
        step = np.linalg.solve(hessian, gradient)
        beta += step
        if np.max(np.abs(step)) < 1e-7:
            break
    return beta


def wilson_interval(p, n, z=1.96):
    den = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / den
    return np.clip(center - half, 0, 1), np.clip(center + half, 0, 1)


def local_wilson_intervals(x_train, y_train, x_query, neighbors=LOCAL_NEIGHBORS):
    """Wilson روی موفقیت های واقعی نزدیک ترین پست های مجموعه آموزش."""
    k = min(neighbors, len(x_train))
    successes, rates, lows, highs, radii = [], [], [], [], []
    for row in np.atleast_2d(x_query):
        distance2 = np.sum((x_train - row) ** 2, axis=1)
        nearest = np.argpartition(distance2, k - 1)[:k]
        success = int(y_train[nearest].sum())
        rate = success / k
        low, high = wilson_interval(rate, k)
        successes.append(success)
        rates.append(rate)
        lows.append(float(low))
        highs.append(float(high))
        radii.append(float(np.sqrt(distance2[nearest].max())))
    return {
        "successes": np.array(successes),
        "n": np.full(len(successes), k),
        "rate": np.array(rates),
        "low": np.array(lows),
        "high": np.array(highs),
        "radius": np.array(radii),
    }


def binary_metrics(y, prediction):
    tp = np.sum((y == 1) & (prediction == 1))
    tn = np.sum((y == 0) & (prediction == 0))
    fp = np.sum((y == 0) & (prediction == 1))
    fn = np.sum((y == 1) & (prediction == 0))
    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0
    return {
        "accuracy": (tp + tn) / len(y), "precision": precision, "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0,
        "true_positive": int(tp), "true_negative": int(tn),
        "false_positive": int(fp), "false_negative": int(fn),
    }


def part5_prediction(posts):
    feature_names = ["s5", "s10", "s20", "s30"]
    x = posts[feature_names].to_numpy(float)
    train_idx, test_idx = random_split(len(posts))
    # تعریف هدف فقط با داده آموزش انجام می شود تا صدک داده آزمون وارد مدل نشود
    target_score_threshold = float(posts.iloc[train_idx]["s1440"].quantile(0.80))
    y = (posts["s1440"] >= target_score_threshold).astype(int).to_numpy()
    x_train0, x_test0 = x[train_idx], x[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    mean = x_train0.mean(axis=0)
    sd = x_train0.std(axis=0, ddof=1)
    x_train = (x_train0 - mean) / sd
    x_test = (x_test0 - mean) / sd
    beta = fit_logistic_by_hand(x_train, y_train)
    train_probability = sigmoid(np.column_stack([np.ones(len(x_train)), x_train]) @ beta)
    # آستانه را فقط با داده آموزش انتخاب می کنیم تا نتیجه آزمون لو نرود.
    choices = np.arange(0.15, 0.71, 0.02)
    train_f1 = [binary_metrics(y_train, (train_probability >= t).astype(int))["f1"] for t in choices]
    threshold = float(choices[int(np.argmax(train_f1))])
    probability = sigmoid(np.column_stack([np.ones(len(x_test)), x_test]) @ beta)
    prediction = (probability >= threshold).astype(int)

    local = local_wilson_intervals(x_train, y_train, x_test)
    low, high = local["low"], local["high"]
    width = high - low
    split_width = np.median(width)
    high_conf = width <= split_width

    metrics = binary_metrics(y_test, prediction)
    metrics.update({
        "train_size": len(train_idx), "test_size": len(test_idx),
        "decision_threshold": threshold, "brier_score": np.mean((probability - y_test) ** 2),
        "target_score_threshold_from_train": target_score_threshold,
        "l2_penalty": L2_PENALTY,
        "local_wilson_neighbors": int(local["n"][0]),
        "relative_confidence_width_median": float(split_width),
        "false_negative_cost_ratio": FN_COST_RATIO,
        "high_confidence_accuracy": np.mean(prediction[high_conf] == y_test[high_conf]),
        "low_confidence_accuracy": np.mean(prediction[~high_conf] == y_test[~high_conf]),
    })
    (TAB / "part5_model_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    test = posts.iloc[test_idx].copy()
    test["actual"] = y_test
    test["probability"] = probability
    test["prediction"] = prediction
    test["local_successes"] = local["successes"]
    test["local_neighbors"] = local["n"]
    test["local_popular_rate"] = local["rate"]
    test["wilson_low"] = low
    test["wilson_high"] = high
    test["ci_width"] = width
    test["confidence_group"] = np.where(high_conf, "relatively_narrow", "relatively_wide")
    test["neighbor_radius"] = local["radius"]
    test["error"] = prediction != y_test
    test["error_type"] = np.select(
        [(prediction == 1) & (y_test == 0), (prediction == 0) & (y_test == 1)],
        ["false_positive", "false_negative"], default="correct",
    )
    test["subscriber_group"] = pd.qcut(test["subscribers"], 3, labels=["small", "medium", "large"])
    test.to_csv(TAB / "part5_test_predictions.csv", index=False)

    calibration = test.assign(bin=pd.cut(test["probability"], np.arange(0, 1.01, 0.1), include_lowest=True))
    calibration = calibration.groupby("bin", observed=False).agg(
        mean_predicted=("probability", "mean"), actual_popular_rate=("actual", "mean"), count=("actual", "size")
    ).dropna().reset_index()
    calibration["bin"] = calibration["bin"].astype(str)
    calibration.to_csv(TAB / "part5_reliability.csv", index=False)

    plt.figure(figsize=(5.4, 5.2))
    plt.plot([0, 1], [0, 1], ls="--", color="gray", label="Perfect calibration")
    plt.plot(calibration["mean_predicted"], calibration["actual_popular_rate"], marker="o", color="#2a6f97")
    for _, row in calibration.iterrows():
        plt.annotate(str(int(row["count"])), (row["mean_predicted"], row["actual_popular_rate"]), fontsize=8)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Observed popular fraction")
    plt.title("Reliability diagram (labels show bin size)")
    plt.grid(alpha=0.2)
    save_plot("part5_reliability_diagram.png")

    error_tables = []
    for col in ["channel", "channel_type", "subscriber_group", "publish_hour"]:
        temp = test.groupby(col, observed=False).agg(error_rate=("error", "mean"), n=("error", "size")).reset_index()
        temp.insert(0, "group_variable", col)
        temp = temp.rename(columns={col: "group"})
        error_tables.append(temp)
    pd.concat(error_tables, ignore_index=True).to_csv(TAB / "part5_error_groups.csv", index=False)

    costs = []
    for threshold in np.arange(0.20, 0.81, 0.05):
        pred = probability >= threshold
        m = binary_metrics(y_test, pred.astype(int))
        costs.append({"threshold": threshold, "false_positive": m["false_positive"],
                      "false_negative": m["false_negative"],
                      "cost_if_FN_is_twice_FP": m["false_positive"] + FN_COST_RATIO * m["false_negative"]})
    pd.DataFrame(costs).to_csv(TAB / "part5_threshold_costs.csv", index=False)

    model = {
        "feature_names": feature_names,
        "mean": mean.tolist(),
        "sd": sd.tolist(),
        "beta": beta.tolist(),
        "decision_threshold": metrics["decision_threshold"],
        "target_score_threshold": target_score_threshold,
        "l2_penalty": L2_PENALTY,
        "local_neighbors": int(local["n"][0]),
        "x_train_standardized": x_train.tolist(),
        "y_train": y_train.astype(int).tolist(),
    }
    (TAB / "part5_saved_model.json").write_text(json.dumps(model, indent=2), encoding="utf-8")
    return metrics, model


def predict_new_post(model, scores):
    scores = np.asarray(scores, dtype=float)
    if scores.shape != (4,):
        raise ValueError("چهار امتیاز برای زمان های 5 و 10 و 20 و 30 دقیقه لازم است")
    mean = np.asarray(model["mean"], dtype=float)
    sd = np.asarray(model["sd"], dtype=float)
    beta = np.asarray(model["beta"], dtype=float)
    row = (scores - mean) / sd
    probability = float(sigmoid(np.r_[1.0, row] @ beta))
    x_train = np.asarray(model["x_train_standardized"], dtype=float)
    y_train = np.asarray(model["y_train"], dtype=int)
    local = local_wilson_intervals(x_train, y_train, row, int(model["local_neighbors"]))
    return {
        "probability": probability,
        "local_popular_rate": float(local["rate"][0]),
        "wilson_low": float(local["low"][0]),
        "wilson_high": float(local["high"][0]),
        "local_successes": int(local["successes"][0]),
        "local_neighbors": int(local["n"][0]),
        "decision": "popular" if probability >= model["decision_threshold"] else "not_popular",
        "decision_threshold": float(model["decision_threshold"]),
    }


def part6_distributions(posts, final_popular):
    rng = np.random.default_rng(SEED)
    ordered = posts.sort_values("published_at").copy()
    y = final_popular.loc[ordered.index].astype(int).to_numpy()
    params = []
    fit_checks = []

    # 1) دوجمله ای: تعداد محبوب ها در بسته های ده تایی
    n = 10
    batch_counts = np.array([y[i:i + n].sum() for i in range(0, len(y) - n + 1, n)])
    p = y.mean()
    k = np.arange(n + 1)
    empirical = pd.Series(batch_counts).value_counts(normalize=True).reindex(k, fill_value=0)
    binomial_pmf = stats.binom.pmf(k, n, p)
    plt.figure(figsize=(6.5, 4.2))
    plt.bar(k - 0.18, empirical, width=0.36, label="Observed batches")
    plt.bar(k + 0.18, binomial_pmf, width=0.36, label="Binomial PMF")
    plt.xlabel("Popular posts in a batch of 10")
    plt.ylabel("Probability")
    plt.title("Binomial fit")
    plt.legend()
    save_plot("part6_binomial.png")
    params += [{"distribution": "binomial", "parameter": "n", "value": n},
               {"distribution": "binomial", "parameter": "p", "value": p}]
    fit_checks.append({"distribution": "binomial", "check": "observed_mean_minus_n_p", "value": batch_counts.mean() - n * p})
    fit_checks.append({"distribution": "binomial", "check": "total_variation_distance", "value": 0.5 * np.abs(empirical - binomial_pmf).sum()})

    # 2) هندسی: چند پست تا محبوب بعدی دیده می شود
    waits, wait = [], 0
    for value in y:
        wait += 1
        if value:
            waits.append(wait)
            wait = 0
    waits = np.array(waits)
    max_k = max(int(waits.max()), int(stats.geom.ppf(0.995, p)))
    gk = np.arange(1, max_k + 1)
    geometric_empirical = pd.Series(waits).value_counts(normalize=True).reindex(gk, fill_value=0)
    geometric_pmf = stats.geom.pmf(gk, p)
    plt.figure(figsize=(6.5, 4.2))
    plt.bar(gk, geometric_empirical, alpha=0.65, label="Observed probability")
    plt.plot(gk, geometric_pmf, "o-", label="Geometric PMF")
    plt.xlabel("Posts until next popular post")
    plt.ylabel("Probability")
    plt.title("Geometric fit")
    plt.legend()
    save_plot("part6_geometric.png")
    params.append({"distribution": "geometric", "parameter": "p", "value": p})
    fit_checks.append({"distribution": "geometric", "check": "observed_mean_minus_1_over_p", "value": waits.mean() - 1 / p})
    geometric_tail = 1 - geometric_pmf.sum()
    fit_checks.append({
        "distribution": "geometric",
        "check": "total_variation_distance_with_tail",
        "value": 0.5 * (np.abs(geometric_empirical - geometric_pmf).sum() + geometric_tail),
    })

    # 3) پواسون: بازدیدهای تازه بین دقیقه 20 و 30
    new_views = (posts["views_30"] - posts["views_20"]).clip(lower=0).astype(int).to_numpy()
    lam = new_views.mean()
    upper = np.percentile(new_views, 97)
    edges = np.unique(np.rint(np.linspace(0, upper, 24)).astype(int))
    if len(edges) < 3:
        edges = np.arange(new_views.max() + 2)
    observed, edges = np.histogram(new_views, bins=edges)
    observed = observed / len(new_views)
    expected = stats.poisson.cdf(edges[1:] - 1, lam) - stats.poisson.cdf(edges[:-1] - 1, lam)
    centers = (edges[:-1] + edges[1:]) / 2
    plt.figure(figsize=(7, 4.2))
    plt.bar(centers, observed, width=np.diff(edges) * 0.85, alpha=0.65, label="Observed")
    plt.plot(centers, expected, "o-", color="#c3423f", label="Poisson probability")
    plt.xlabel("New views from 20 to 30 min")
    plt.ylabel("Probability per bin")
    plt.title("Poisson fit")
    plt.legend()
    save_plot("part6_poisson.png")
    params += [{"distribution": "poisson", "parameter": "lambda", "value": lam},
               {"distribution": "poisson", "parameter": "variance_to_mean", "value": new_views.var(ddof=1) / lam}]
    fit_checks.append({"distribution": "poisson", "check": "variance_to_mean", "value": new_views.var(ddof=1) / lam})

    # 4) فوق هندسی: انتخاب بیست پست بدون جایگذاری
    population_n, popular_k = len(y), int(y.sum())
    sample_n = min(20, population_n // 4)
    sampled_counts = [rng.choice(y, size=sample_n, replace=False).sum() for _ in range(2500)]
    hk = np.arange(sample_n + 1)
    h_emp = pd.Series(sampled_counts).value_counts(normalize=True).reindex(hk, fill_value=0)
    hypergeometric_pmf = stats.hypergeom.pmf(hk, population_n, popular_k, sample_n)
    plt.figure(figsize=(6.5, 4.2))
    plt.bar(hk - 0.18, h_emp, width=0.36, label="Repeated samples")
    plt.bar(hk + 0.18, hypergeometric_pmf, width=0.36, label="Hypergeometric PMF")
    plt.xlabel("Popular posts in sample")
    plt.ylabel("Probability")
    plt.title("Hypergeometric fit")
    plt.legend()
    save_plot("part6_hypergeometric.png")
    params += [{"distribution": "hypergeometric", "parameter": "N", "value": population_n},
               {"distribution": "hypergeometric", "parameter": "K", "value": popular_k},
               {"distribution": "hypergeometric", "parameter": "n", "value": sample_n}]
    fit_checks.append({"distribution": "hypergeometric", "check": "observed_mean_minus_nK_over_N", "value": np.mean(sampled_counts) - sample_n * popular_k / population_n})
    fit_checks.append({
        "distribution": "hypergeometric",
        "check": "total_variation_distance",
        "value": 0.5 * np.abs(h_emp - hypergeometric_pmf).sum(),
    })

    # 5) نمایی: فاصله زمانی پست های محبوب
    popular_times = ordered.loc[y.astype(bool), "published_at"].sort_values()
    wait_hours = popular_times.diff().dropna().dt.total_seconds().to_numpy() / 3600
    scale = wait_hours.mean()
    x = np.linspace(0, np.percentile(wait_hours, 97), 250)
    plt.figure(figsize=(6.5, 4.2))
    plt.hist(wait_hours, bins=15, density=True, alpha=0.65, label="Observed")
    plt.plot(x, stats.expon.pdf(x, scale=scale), color="#c3423f", lw=2, label="Exponential PDF")
    plt.xlabel("Hours between popular posts")
    plt.ylabel("Density")
    plt.title("Exponential fit")
    plt.legend()
    save_plot("part6_exponential.png")
    params.append({"distribution": "exponential", "parameter": "scale_mean_hours", "value": scale})
    exp_ks = stats.kstest(wait_hours, "expon", args=(0, scale))
    fit_checks.append({"distribution": "exponential", "check": "KS_p_value", "value": exp_ks.pvalue})

    # 6) نرمال برای امتیاز نهایی
    score = posts["s1440"].to_numpy()
    mu, sigma = score.mean(), score.std(ddof=1)
    params += [{"distribution": "normal", "parameter": "mu", "value": mu},
               {"distribution": "normal", "parameter": "sigma", "value": sigma}]
    normal_test = stats.normaltest(score)
    fit_checks.append({"distribution": "normal", "check": "D_Agostino_p_value", "value": normal_test.pvalue})
    nx = np.linspace(score.min(), score.max(), 300)
    plt.figure(figsize=(6.5, 4.2))
    plt.hist(score, bins=16, density=True, alpha=0.65, label="Observed score")
    plt.plot(nx, stats.norm.pdf(nx, mu, sigma), color="#c3423f", lw=2, label="Normal PDF")
    plt.xlabel("Popularity score at 24h")
    plt.ylabel("Density")
    plt.title("Normal fit")
    plt.legend()
    save_plot("part6_normal.png")

    # 7) قضیه حد مرکزی
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    raw_views = posts["views_1440"].to_numpy()
    for ax, sample_size in zip(axes.flat, [5, 15, 30, 100]):
        means = rng.choice(raw_views, size=(1000, sample_size), replace=True).mean(axis=1)
        ax.hist(means, bins=25, density=True, alpha=0.7, color="#4d7ea8")
        mx, sx = means.mean(), means.std(ddof=1)
        gx = np.linspace(means.min(), means.max(), 200)
        ax.plot(gx, stats.norm.pdf(gx, mx, sx), color="#c3423f", lw=1.5)
        ax.set_title(f"n={sample_size}")
        ax.set_xlabel("Sample mean of views")
    fig.suptitle("Central Limit Theorem: 1000 sample means")
    save_plot("part6_clt.png")

    # 8) توزیع مشترک، حاشیه ای و شرطی
    x_bin = pd.qcut(posts["s20"], 4, labels=["X1", "X2", "X3", "X4"])
    y_bin = pd.qcut(posts["s1440"], 4, labels=["Y1", "Y2", "Y3", "Y4"])
    joint = pd.crosstab(x_bin, y_bin, normalize="all")
    conditional = pd.crosstab(x_bin, y_bin, normalize="index")
    x_marginal = joint.sum(axis=1).rename("P_X")
    y_marginal = joint.sum(axis=0).rename("P_Y")
    joint.to_csv(TAB / "part6_joint_distribution.csv")
    conditional.to_csv(TAB / "part6_conditional_Y_given_X.csv")
    x_marginal.to_csv(TAB / "part6_marginal_X.csv")
    y_marginal.to_csv(TAB / "part6_marginal_Y.csv")
    checks = {"joint_sum": joint.to_numpy().sum(), "x_marginal_sum": x_marginal.sum(), "y_marginal_sum": y_marginal.sum()}
    (TAB / "part6_probability_checks.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")

    pd.DataFrame(params).to_csv(TAB / "part6_distribution_parameters.csv", index=False)
    pd.DataFrame(fit_checks).to_csv(TAB / "part6_distribution_fit_checks.csv", index=False)


def main(filename):
    FIG.mkdir(parents=True, exist_ok=True)
    TAB.mkdir(parents=True, exist_ok=True)
    data = load_and_clean(filename)
    scored = add_popularity_score(data)
    posts = make_wide(scored)
    scored.to_csv(TAB / "clean_scored_snapshots.csv", index=False)
    posts.to_csv(TAB / "post_scores_wide.csv", index=False)

    desc = part1_descriptive(posts)
    corr = part2_early_prediction(posts)
    _, final_popular, hypothesis = part3_stability(posts)
    part4_bayes(posts, final_popular)
    model_metrics, saved_model = part5_prediction(posts)
    part6_distributions(posts, final_popular)

    summary = [
        f"posts={len(posts)}",
        f"channels={posts['channel'].nunique()}",
        f"score_mean={desc['arithmetic_mean']:.3f}",
        f"score_skewness={desc['skewness']:.3f}",
        f"first_substantial_time={corr.loc[corr['substantial_abs_r_ge_0.6'], 'time_min'].min() if corr['substantial_abs_r_ge_0.6'].any() else 'none'}",
        f"stability_p_value={hypothesis['p_value']:.6g}",
        f"model_accuracy={model_metrics['accuracy']:.3f}",
        f"model_f1={model_metrics['f1']:.3f}",
    ]
    (OUT / "quick_summary.txt").write_text("\n".join(summary), encoding="utf-8")
    print(f"تحلیل تمام شد. جدول ها در {TAB} و نمودارها در {FIG} هستند.")
    return saved_model


def print_prediction(result):
    decision = "محبوب" if result["decision"] == "popular" else "غیر محبوب"
    print("\nپیش بینی پست جدید")
    print(f"احتمال لجستیک: {100 * result['probability']:.2f} درصد")
    print(
        f"Wilson محلی: {100 * result['wilson_low']:.2f} تا "
        f"{100 * result['wilson_high']:.2f} درصد  "
        f"بر اساس {result['local_successes']} موفقیت از {result['local_neighbors']} همسایه"
    )
    print(f"تصمیم نهایی: {decision}  آستانه {result['decision_threshold']:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_file", nargs="?", default=None)
    parser.add_argument(
        "--predict", nargs=4, type=float, metavar=("S5", "S10", "S20", "S30"),
        help="بعد از تحلیل برای چهار امتیاز یک پست جدید پیش بینی چاپ می کند",
    )
    args = parser.parse_args()
    # داخل پروژه داده در data است ولی داخل بسته تحویل کنار فایل کد قرار می گیرد
    if args.csv_file is None:
        if Path("data/telegram_snapshots.csv").exists():
            args.csv_file = "data/telegram_snapshots.csv"
        else:
            args.csv_file = "telegram_snapshots.csv"
    fitted_model = main(args.csv_file)
    if args.predict is not None:
        print_prediction(predict_new_post(fitted_model, args.predict))
