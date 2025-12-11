"""
Streamlit app for N-BEATS production forecasting with uncertainty (SPE wells).

Tabs:
1) Single-well forecast (MC-dropout; rate + cumulative with P10/P50/P90 vs observed)
2) DL Type curves (Vo DL type curves for Low/Medium/High EUR groups with
   P10–P90 bands, optional observed-mean overlay, and download buttons)

You can either:
- use the default 'spe_all_wells.csv', or
- upload another CSV with columns:
    well_name, time_days, qg_mmscf_day, pwf_psi

Model:
- nbeats_uncert_model.pt  (trained N-BEATS with dropout)

Run:
    streamlit run app_nbeats_streamlit.py
"""

import os
import io
import random
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import streamlit as st
import matplotlib.pyplot as plt
from torch.utils.data import Dataset

# ---------------------- CONFIG ---------------------- #

DEFAULT_CSV_PATH = "spe_all_wells.csv"
MODEL_PATH = "nbeats_uncert_model.pt"

INPUT_LEN = 90
OUTPUT_LEN = 30
STRIDE = 10

DROPOUT_P = 0.2          # must match training
N_MC_SINGLE_DEFAULT = 100
N_MC_TYPE_DEFAULT = 100

SEED = 2025
USE_CUDA = False         # use CPU for Streamlit to avoid GPU issues

# feature indices in [t_norm, log_t_norm, pwf_norm, q_norm, q0_norm, t0_norm]
IDX_TNORM = 0
IDX_LOGT = 1
IDX_PWF = 2
IDX_Q = 3
IDX_Q0 = 4
IDX_T0 = 5

FEATURE_IDX = [IDX_Q, IDX_PWF, IDX_LOGT]   # used features (same as training)


# -------------------- UTILITIES --------------------- #

def set_global_seed(seed: int = 2025):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class MultiStepWindowDataset(Dataset):
    """
    Build sliding windows for multi-step forecasting.

    Each sample:
        x: (INPUT_LEN, 6)   -> [t_norm, log_t_norm, pwf_norm, q_norm, q0_norm, t0_norm]
        y: (OUTPUT_LEN,)    -> future q_norm
        time: (INPUT_LEN+OUTPUT_LEN,) absolute day
        well_name: str
        start_idx: int
    """

    def __init__(
        self,
        df,
        max_time,
        pwf_mean, pwf_std,
        q_mean, q_std,
        input_len=INPUT_LEN,
        output_len=OUTPUT_LEN,
        stride=STRIDE,
    ):
        super().__init__()
        self.samples = []
        self.input_len = input_len
        self.output_len = output_len
        self.stride = stride

        self.max_time = max_time
        self.pwf_mean = pwf_mean
        self.pwf_std = max(pwf_std, 1e-6)
        self.q_mean = q_mean
        self.q_std = max(q_std, 1e-6)

        df = df.sort_values(["well_name", "time_days"]).reset_index(drop=True)

        for wname, g in df.groupby("well_name"):
            g = g.sort_values("time_days")
            T = len(g)
            if T < input_len + output_len + 1:
                continue

            time = g["time_days"].values.astype("float32")
            pwf = g["pwf_psi"].values.astype("float32")
            qg = g["qg_mmscf_day"].values.astype("float32")

            t_norm = time / max_time
            log_t = np.log1p(time)
            log_t_norm = log_t / (log_t.max() if log_t.max() > 0 else 1.0)

            pwf_norm = (pwf - self.pwf_mean) / self.pwf_std
            q_norm = (qg - self.q_mean) / self.q_std

            q0_norm = (qg[0] - self.q_mean) / self.q_std
            t0_norm = time[0] / max_time
            q0_feat = np.full_like(time, q0_norm, dtype="float32")
            t0_feat = np.full_like(time, t0_norm, dtype="float32")

            feats = np.stack(
                [t_norm, log_t_norm, pwf_norm, q_norm, q0_feat, t0_feat],
                axis=-1,
            )
            target_series = q_norm

            start = 0
            while start + input_len + output_len <= T:
                x = feats[start:start + input_len]
                y = target_series[start + input_len:
                                  start + input_len + output_len]
                self.samples.append(
                    {
                        "x": torch.tensor(x, dtype=torch.float32),
                        "y": torch.tensor(y, dtype=torch.float32),
                        "well_name": wname,
                        "start_idx": start,
                        "time": torch.tensor(
                            time[start:start + input_len + output_len],
                            dtype=torch.float32,
                        ),
                    }
                )
                start += stride

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class NBeatsDropoutBlock(nn.Module):
    def __init__(self, input_dim, output_len,
                 hidden_dim=128, n_layers=4, dropout_p=0.2):
        super().__init__()
        self.input_dim = input_dim
        self.output_len = output_len

        layers = []
        in_dim = input_dim
        for _ in range(n_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_p))
            in_dim = hidden_dim
        self.fc = nn.Sequential(*layers)

        self.backcast_layer = nn.Linear(hidden_dim, input_dim)
        self.forecast_layer = nn.Linear(hidden_dim, output_len)

    def forward(self, x):
        h = self.fc(x)
        backcast = self.backcast_layer(h)
        forecast = self.forecast_layer(h)
        return backcast, forecast


class NBeatsDropout(nn.Module):
    def __init__(
        self,
        input_len,
        output_len,
        feature_idx,
        n_stacks=2,
        n_blocks_per_stack=3,
        hidden_dim=128,
        n_layers=4,
        dropout_p=0.2,
    ):
        super().__init__()
        self.input_len = input_len
        self.output_len = output_len
        self.feature_idx = feature_idx

        self.n_features_used = len(feature_idx)
        self.input_dim = input_len * self.n_features_used

        blocks = []
        for _ in range(n_stacks):
            for _ in range(n_blocks_per_stack):
                blocks.append(
                    NBeatsDropoutBlock(
                        input_dim=self.input_dim,
                        output_len=output_len,
                        hidden_dim=hidden_dim,
                        n_layers=n_layers,
                        dropout_p=dropout_p,
                    )
                )
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x):
        # x: (B, T, 6)
        x_sel = x[:, :, self.feature_idx]      # (B, T, F_sel)
        backcast = x_sel.reshape(x_sel.size(0), -1)  # flatten

        forecast_total = 0.0
        for block in self.blocks:
            b, f = block(backcast)
            backcast = backcast - b
            forecast_total = forecast_total + f
        return forecast_total                 # (B, OUTPUT_LEN)


# -------------------- MC DROPOUT HELPERS -------------------- #

def mc_dropout_envelopes_for_well(
    model,
    dataset,
    df_full,
    well_name,
    device,
    q_mean,
    q_std,
    n_mc_samples=100,
):
    """Run MC-dropout for a single well, aggregate by production day."""
    well_samples = [s for s in dataset.samples if s["well_name"] == well_name]
    if len(well_samples) == 0:
        return None

    agg_pred = defaultdict(list)
    model.train()  # enable dropout
    with torch.no_grad():
        for sample in well_samples:
            x = sample["x"].unsqueeze(0).to(device)
            time_full = sample["time"].cpu().numpy()
            t_out = time_full[INPUT_LEN:]

            preds_mc = []
            for _ in range(n_mc_samples):
                y_pred_norm = model(x).cpu().numpy()[0]   # (H,)
                preds_mc.append(y_pred_norm)
            preds_mc = np.stack(preds_mc, axis=0)         # (Nmc, H)
            preds_mc = preds_mc * q_std + q_mean          # de-normalise

            for j, t_day in enumerate(t_out):
                day = int(round(float(t_day)))
                agg_pred[day].append(preds_mc[:, j])

    days_sorted = sorted(agg_pred.keys())
    p10_list, p50_list, p90_list = [], [], []

    for d in days_sorted:
        arr = np.concatenate(agg_pred[d], axis=0)
        p10_list.append(np.quantile(arr, 0.10))
        p50_list.append(np.quantile(arr, 0.50))
        p90_list.append(np.quantile(arr, 0.90))

    time_days = np.array(days_sorted, dtype="float32")
    p10 = np.array(p10_list)
    p50 = np.array(p50_list)
    p90 = np.array(p90_list)

    # observed rates at those days
    g = (
        df_full[df_full["well_name"] == well_name]
        .sort_values("time_days")
        .reset_index(drop=True)
    )
    obs_time = g["time_days"].values.astype("float32")
    obs_rate = g["qg_mmscf_day"].values.astype("float32")
    obs_dict = {int(round(t)): q for t, q in zip(obs_time, obs_rate)}
    q_obs = np.array([obs_dict.get(d, np.nan) for d in days_sorted],
                     dtype="float32")

    def cumulative(q):
        q_filled = np.nan_to_num(q, nan=0.0)
        return np.cumsum(q_filled)

    cum_p10 = cumulative(p10)
    cum_p50 = cumulative(p50)
    cum_p90 = cumulative(p90)
    cum_obs = cumulative(q_obs)

    return {
        "time_days": time_days,
        "p10": p10,
        "p50": p50,
        "p90": p90,
        "q_obs": q_obs,
        "cum_p10": cum_p10,
        "cum_p50": cum_p50,
        "cum_p90": cum_p90,
        "cum_obs": cum_obs,
    }


def mc_dropout_all_wells(
    model, dataset, df_full, device, stats, n_mc_samples=100
):
    """
    Run MC-dropout for ALL wells and compute EUR P10/P50/P90 per well.

    Returns:
        mc_results: {well_name: res_dict}
        eur_df: DataFrame with EUR per well
    """
    results = {}
    eur_rows = []

    wells = sorted(df_full["well_name"].unique().tolist())
    for w in wells:
        res = mc_dropout_envelopes_for_well(
            model=model,
            dataset=dataset,
            df_full=df_full,
            well_name=w,
            device=device,
            q_mean=stats["q_mean"],
            q_std=stats["q_std"],
            n_mc_samples=n_mc_samples,
        )
        if res is None:
            continue

        results[w] = res
        EUR_P10 = float(res["cum_p10"][-1])
        EUR_P50 = float(res["cum_p50"][-1])
        EUR_P90 = float(res["cum_p90"][-1])
        eur_rows.append(
            {
                "well_name": w,
                "EUR_P10": EUR_P10,
                "EUR_P50": EUR_P50,
                "EUR_P90": EUR_P90,
            }
        )

    eur_df = pd.DataFrame(eur_rows)
    return results, eur_df


def build_vo_typecurve_group(
    mc_results: dict,
    well_names: list,
    n_points: int = 200,
):
    """
    Build Vo DL type curve for a set of wells using their MC P50 trajectories.

    Also computes the **mean observed rate + cumulative** across wells,
    aligned in relative time.

    Returns dict with keys:
        t_grid, p10, p50, p90, cum_p10, cum_p50, cum_p90,
        obs_mean_rate, obs_mean_cum
    """
    if len(well_names) == 0:
        return None

    # Determine max relative time shared by all wells in the group
    t_max_list = []
    for w in well_names:
        res = mc_results[w]
        t_rel = res["time_days"] - res["time_days"][0]
        t_max_list.append(t_rel[-1])
    t_max_group = float(min(t_max_list))

    t_grid = np.linspace(0.0, t_max_group, n_points)

    p50_mat = []
    obs_mat = []

    for w in well_names:
        res = mc_results[w]
        t_rel = res["time_days"] - res["time_days"][0]
        p50_interp = np.interp(t_grid, t_rel, res["p50"])
        obs_interp = np.interp(t_grid, t_rel, res["q_obs"])
        p50_mat.append(p50_interp)
        obs_mat.append(obs_interp)

    p50_mat = np.stack(p50_mat, axis=0)  # (n_wells, n_points)
    obs_mat = np.stack(obs_mat, axis=0)

    group_p50 = np.median(p50_mat, axis=0)
    group_p10 = np.quantile(p50_mat, 0.10, axis=0)
    group_p90 = np.quantile(p50_mat, 0.90, axis=0)

    group_obs_mean = np.mean(obs_mat, axis=0)

    def cumulative(q):
        return np.cumsum(np.maximum(q, 0.0))  # simple positive integration

    group_cum_p50 = cumulative(group_p50)
    group_cum_p10 = cumulative(group_p10)
    group_cum_p90 = cumulative(group_p90)
    group_cum_obs = cumulative(group_obs_mean)

    return {
        "t_grid": t_grid,
        "p10": group_p10,
        "p50": group_p50,
        "p90": group_p90,
        "cum_p10": group_cum_p10,
        "cum_p50": group_cum_p50,
        "cum_p90": group_cum_p90,
        "obs_mean_rate": group_obs_mean,
        "obs_mean_cum": group_cum_obs,
    }


def fig_to_png_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
    buf.seek(0)
    return buf.getvalue()


# -------------------- DATA / MODEL LOADERS -------------------- #

def load_dataframe(default_path: str, uploaded_file):
    """
    Load and prepare dataframe from default path or uploaded CSV.
    """
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        source = "uploaded CSV"
    else:
        if not os.path.exists(default_path):
            raise FileNotFoundError(f"CSV not found at {default_path}")
        df = pd.read_csv(default_path)
        source = default_path

    needed_cols = ["well_name", "time_days", "qg_mmscf_day", "pwf_psi"]
    for c in needed_cols:
        if c not in df.columns:
            raise ValueError(
                f"Column '{c}' not found in CSV. Required columns: {needed_cols}. "
                f"Got: {df.columns.tolist()}"
            )

    df = df.sort_values(["well_name", "time_days"]).reset_index(drop=True)

    # interpolate basic missing values
    df[["qg_mmscf_day", "pwf_psi"]] = (
        df.groupby("well_name")[["qg_mmscf_day", "pwf_psi"]]
          .transform(lambda g: g.interpolate().ffill().bfill())
    )
    return df, source


def compute_train_stats(df: pd.DataFrame):
    """
    Approximate training split to compute normalisation statistics.
    """
    set_global_seed(SEED)

    all_wells = sorted(df["well_name"].unique().tolist())
    random.shuffle(all_wells)
    split = int(0.8 * len(all_wells))
    train_wells = set(all_wells[:split])

    df_train = df[df["well_name"].isin(train_wells)].copy()

    max_time = float(df["time_days"].max())
    pwf_mean = float(df_train["pwf_psi"].mean())
    pwf_std = float(df_train["pwf_psi"].std())
    q_mean = float(df_train["qg_mmscf_day"].mean())
    q_std = float(df_train["qg_mmscf_day"].std())

    return {
        "max_time": max_time,
        "pwf_mean": pwf_mean,
        "pwf_std": pwf_std,
        "q_mean": q_mean,
        "q_std": q_std,
    }


def build_dataset(df: pd.DataFrame, stats: dict):
    return MultiStepWindowDataset(
        df,
        max_time=stats["max_time"],
        pwf_mean=stats["pwf_mean"],
        pwf_std=stats["pwf_std"],
        q_mean=stats["q_mean"],
        q_std=stats["q_std"],
        input_len=INPUT_LEN,
        output_len=OUTPUT_LEN,
        stride=STRIDE,
    )


def load_model(device: torch.device):
    model = NBeatsDropout(
        input_len=INPUT_LEN,
        output_len=OUTPUT_LEN,
        feature_idx=FEATURE_IDX,
        n_stacks=2,
        n_blocks_per_stack=3,
        hidden_dim=128,
        n_layers=4,
        dropout_p=DROPOUT_P,
    ).to(device)

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file '{MODEL_PATH}' not found. "
            f"Train the model first and save it with this name."
        )

    state = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state)
    return model


# -------------------------- MAIN APP -------------------------- #

def main():
    st.title("N-BEATS DL Forecasting & Vo DL Type Curves (SPE wells)")

    st.markdown(
        """
        This app uses a trained **N-BEATS** model with MC Dropout to:

        - Predict gas **rate & cumulative** for individual wells with **P10/P50/P90** bands  
        - Build **Vo DL type curves** for **Low / Medium / High EUR groups** with uncertainty envelopes

        You can use the default SPE CSV or upload your own CSV with the same columns.
        """
    )

    set_global_seed(SEED)
    if USE_CUDA and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # ---------- SIDEBAR: DATASET + CONTROLS ---------- #
    st.sidebar.header("Dataset & controls")

    uploaded_file = st.sidebar.file_uploader(
        "Upload CSV (optional; must have well_name, time_days, qg_mmscf_day, pwf_psi)",
        type=["csv"],
    )

    df, data_source = load_dataframe(DEFAULT_CSV_PATH, uploaded_file)
    st.sidebar.caption(f"Using data: **{data_source}**")

    stats = compute_train_stats(df)
    dataset = build_dataset(df, stats)
    model = load_model(device)

    all_wells = sorted(df["well_name"].unique().tolist())

    well_name = st.sidebar.selectbox(
        "Single-well forecast – choose well", all_wells
    )
    n_mc_single = st.sidebar.slider(
        "MC samples (single-well)",
        min_value=20,
        max_value=200,
        value=N_MC_SINGLE_DEFAULT,
        step=10,
    )
    run_single = st.sidebar.button("Run single-well forecast")

    st.sidebar.markdown("---")
    n_mc_type = st.sidebar.slider(
        "MC samples (type curves)",
        min_value=20,
        max_value=200,
        value=N_MC_TYPE_DEFAULT,
        step=10,
    )
    show_obs_tc = st.sidebar.checkbox(
        "Show observed average on type curves", value=True
    )
    run_type = st.sidebar.button("Recompute DL type curves")

    tab1, tab2 = st.tabs(["Single-well forecast", "DL Type curves"])

    # ------- TAB 1: SINGLE WELL FORECAST ------- #
    with tab1:
        st.subheader("Single-well N-BEATS forecast with uncertainty")

        if run_single:
            with st.spinner(
                f"Running MC dropout ({n_mc_single} samples) for {well_name}..."
            ):
                res = mc_dropout_envelopes_for_well(
                    model=model,
                    dataset=dataset,
                    df_full=df,
                    well_name=well_name,
                    device=device,
                    q_mean=stats["q_mean"],
                    q_std=stats["q_std"],
                    n_mc_samples=n_mc_single,
                )

            if res is None:
                st.error(f"No samples could be built for well {well_name}.")
            else:
                time_days = res["time_days"]
                q_obs = res["q_obs"]
                p10, p50, p90 = res["p10"], res["p50"], res["p90"]
                cum_obs = res["cum_obs"]
                cum_p10, cum_p50, cum_p90 = (
                    res["cum_p10"],
                    res["cum_p50"],
                    res["cum_p90"],
                )

                # Rate plot
                fig_rate, ax = plt.subplots(figsize=(9, 4))
                ax.plot(time_days, q_obs, label="Observed",
                        linewidth=1.5, color="black")
                ax.plot(time_days, p50, label="N-BEATS P50", linestyle="--")
                ax.fill_between(time_days, p10, p90, alpha=0.3,
                                label="P10–P90 band")
                ax.set_title(f"Well {well_name}: Rate – N-BEATS with uncertainty")
                ax.set_xlabel("Time (days)")
                ax.set_ylabel("Gas rate (MMscf/day)")
                ax.grid(True, alpha=0.3)
                ax.legend()
                fig_rate.tight_layout()
                st.pyplot(fig_rate)

                # Cumulative plot
                fig_cum, ax2 = plt.subplots(figsize=(9, 4))
                ax2.plot(time_days, cum_obs, label="Observed",
                         linewidth=1.5, color="black")
                ax2.plot(time_days, cum_p50, label="N-BEATS P50",
                         linestyle="--")
                ax2.fill_between(time_days, cum_p10, cum_p90, alpha=0.3,
                                 label="P10–P90 band")
                ax2.set_title(
                    f"Well {well_name}: Cumulative – N-BEATS with uncertainty"
                )
                ax2.set_xlabel("Time (days)")
                ax2.set_ylabel("Cumulative gas (MMscf)")
                ax2.grid(True, alpha=0.3)
                ax2.legend()
                fig_cum.tight_layout()
                st.pyplot(fig_cum)

                # EUR summary
                EUR_P10 = float(cum_p10[-1])
                EUR_P50 = float(cum_p50[-1])
                EUR_P90 = float(cum_p90[-1])
                st.markdown(
                    f"**EUR from DL uncertainty bands** for `{well_name}` "
                    f"(MMscf):  **P10 = {EUR_P10:,.0f}**, "
                    f"**P50 = {EUR_P50:,.0f}**,  "
                    f"**P90 = {EUR_P90:,.0f}**"
                )
        else:
            st.info(
                "Use the sidebar to select a well, choose MC samples, "
                "and click **Run single-well forecast**."
            )

    # ------- TAB 2: DL TYPE CURVES ------- #
    with tab2:
        st.subheader("Vo DL Type curves by EUR group (N-BEATS P50 with uncertainty)")

        st.markdown(
            """
            This tab:
            1. Runs MC-dropout N-BEATS predictions for **all wells**.  
            2. Computes **EUR P10/P50/P90** per well.  
            3. Sorts wells by EUR_P50 and splits into **Low / Medium / High** terciles.  
            4. Builds **Vo DL type curves** (median P50 across wells) + **P10–P90 bands**,  
               with optional **observed average rate/cumulative** overlay.
            """
        )

        if run_type:
            with st.spinner(
                f"Running MC dropout ({n_mc_type} samples) for ALL wells "
                "and building type curves..."
            ):
                mc_results, eur_df = mc_dropout_all_wells(
                    model=model,
                    dataset=dataset,
                    df_full=df,
                    device=device,
                    stats=stats,
                    n_mc_samples=n_mc_type,
                )

            if eur_df.empty:
                st.error("Could not compute EUR for any well.")
            else:
                # sort wells by EUR_P50 and split into terciles
                eur_df_sorted = eur_df.sort_values("EUR_P50").reset_index(drop=True)
                n = len(eur_df_sorted)
                n_low = n // 3
                n_med = n // 3
                low_wells = eur_df_sorted.iloc[:n_low]["well_name"].tolist()
                med_wells = eur_df_sorted.iloc[n_low:n_low + n_med]["well_name"].tolist()
                high_wells = eur_df_sorted.iloc[n_low + n_med:]["well_name"].tolist()

                # Build type curves
                high_tc = build_vo_typecurve_group(mc_results, high_wells)
                med_tc = build_vo_typecurve_group(mc_results, med_wells)
                low_tc = build_vo_typecurve_group(mc_results, low_wells)

                # EUR summary table (mean values per group)
                group_rows = []
                for name, wells in [("Low", low_wells),
                                    ("Medium", med_wells),
                                    ("High", high_wells)]:
                    sub = eur_df[eur_df["well_name"].isin(wells)]
                    if len(sub) == 0:
                        continue
                    group_rows.append(
                        {
                            "Group": name,
                            "n_wells": len(sub),
                            "Mean_EUR_P10": sub["EUR_P10"].mean(),
                            "Mean_EUR_P50": sub["EUR_P50"].mean(),
                            "Mean_EUR_P90": sub["EUR_P90"].mean(),
                        }
                    )
                st.markdown("### EUR summary per group (MMscf)")
                eur_group_df = pd.DataFrame(group_rows)
                st.dataframe(
                    eur_group_df.style.format(
                        {"Mean_EUR_P10": "{:,.0f}",
                         "Mean_EUR_P50": "{:,.0f}",
                         "Mean_EUR_P90": "{:,.0f}"}
                    ),
                    use_container_width=True,
                )

                # plotting helper
                def plot_group_tc(tc, title_prefix, show_observed=False):
                    if tc is None:
                        return None
                    t = tc["t_grid"]
                    p10, p50, p90 = tc["p10"], tc["p50"], tc["p90"]
                    c10, c50, c90 = tc["cum_p10"], tc["cum_p50"], tc["cum_p90"]
                    obs_r = tc["obs_mean_rate"]
                    obs_c = tc["obs_mean_cum"]

                    fig, (ax1, ax2) = plt.subplots(
                        2, 1, figsize=(9, 6), sharex=True
                    )

                    # Rate
                    ax1.plot(t, p50, label="DL type curve P50", color="C0")
                    ax1.fill_between(
                        t, p10, p90, color="C0", alpha=0.2, label="DL P10–P90 band"
                    )
                    if show_observed:
                        ax1.plot(
                            t, obs_r,
                            label="Observed mean rate",
                            color="black",
                            linewidth=1.3,
                            alpha=0.8,
                        )
                    ax1.set_ylabel("Gas rate (MMscf/day)")
                    ax1.set_title(f"{title_prefix} – Rate")
                    ax1.grid(True, alpha=0.3)
                    ax1.legend()

                    # Cumulative
                    ax2.plot(t, c50, label="DL type curve P50", color="C1")
                    ax2.fill_between(
                        t, c10, c90, color="C1", alpha=0.2,
                        label="DL P10–P90 band"
                    )
                    if show_observed:
                        ax2.plot(
                            t, obs_c,
                            label="Observed mean cumulative",
                            color="black",
                            linewidth=1.3,
                            alpha=0.8,
                        )
                    ax2.set_xlabel("Time since prediction start (days)")
                    ax2.set_ylabel("Cumulative gas (MMscf)")
                    ax2.set_title(f"{title_prefix} – Cumulative")
                    ax2.grid(True, alpha=0.3)
                    ax2.legend()

                    fig.tight_layout()
                    return fig

                # Plot High / Medium / Low groups (Vo DL type curves)
                fig_high = fig_med = fig_low = None

                if high_tc is not None:
                    fig_high = plot_group_tc(
                        high_tc,
                        title_prefix="Vo DL Type Curve – High EUR Group",
                        show_observed=show_obs_tc,
                    )
                    st.pyplot(fig_high)

                if med_tc is not None:
                    fig_med = plot_group_tc(
                        med_tc,
                        title_prefix="Vo DL Type Curve – Medium EUR Group",
                        show_observed=show_obs_tc,
                    )
                    st.pyplot(fig_med)

                if low_tc is not None:
                    fig_low = plot_group_tc(
                        low_tc,
                        title_prefix="Vo DL Type Curve – Low EUR Group",
                        show_observed=show_obs_tc,
                    )
                    st.pyplot(fig_low)

                # -------- DOWNLOAD BUTTONS (FIGURES + TABLES) -------- #
                st.markdown("### Download figures & EUR tables")

                # EUR tables
                eur_group_csv = eur_group_df.to_csv(index=False).encode("utf-8")
                eur_all_csv = (
                    eur_df.sort_values("EUR_P50", ascending=False)
                          .to_csv(index=False)
                          .encode("utf-8")
                )

                st.download_button(
                    "Download EUR group summary (CSV)",
                    data=eur_group_csv,
                    file_name="eur_group_summary.csv",
                    mime="text/csv",
                )
                st.download_button(
                    "Download per-well EUR table (CSV)",
                    data=eur_all_csv,
                    file_name="eur_per_well.csv",
                    mime="text/csv",
                )

                # Figures
                if fig_high is not None:
                    st.download_button(
                        "Download High EUR type curve figure (PNG)",
                        data=fig_to_png_bytes(fig_high),
                        file_name="vo_DL_typecurve_High.png",
                        mime="image/png",
                    )
                if fig_med is not None:
                    st.download_button(
                        "Download Medium EUR type curve figure (PNG)",
                        data=fig_to_png_bytes(fig_med),
                        file_name="vo_DL_typecurve_Medium.png",
                        mime="image/png",
                    )
                if fig_low is not None:
                    st.download_button(
                        "Download Low EUR type curve figure (PNG)",
                        data=fig_to_png_bytes(fig_low),
                        file_name="vo_DL_typecurve_Low.png",
                        mime="image/png",
                    )

        else:
            st.info(
                "Use the sidebar to choose MC samples and click "
                "**Recompute DL type curves** to build Low/Medium/High EUR "
                "Vo DL type curves with uncertainty bands. "
                "You can also toggle the observed average overlay."
            )


if __name__ == "__main__":
    main()
