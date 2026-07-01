"""
src/data/regimes.py

Assigns each month in the macro panel to one of three market regimes
based on VIX level. Uses well-established practitioner thresholds:

    Regime 0 — Calm:      VIX < 20
    Regime 1 — Stressed:  20 <= VIX < 30
    Regime 2 — Crisis:    VIX >= 30

Uses the RAW UNITS panel (macro_panel_raw_units.csv), not the scaled
panel, because VIX thresholds are defined in real VIX units. A scaled
VIX value of -0.34 is meaningless — a raw VIX of 45 is not.
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os


# Define thresholds as module-level constants so they appear in one
# place only. If you ever want to experiment with different cutoffs,
# you change them here and every downstream function picks up the
# change automatically.
VIX_CALM_THRESHOLD    = 20   # below this = calm
VIX_CRISIS_THRESHOLD  = 30   # above this = crisis
                              # between the two = stressed


def assign_regimes(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Assigns a regime label to each row based on the VIX column.

    Args:
        panel: the raw-units macro panel, must contain a 'VIX' column
               with actual VIX index levels (not scaled values)

    Returns:
        The same panel with two new columns appended:
            'regime'      — integer label: 0, 1, or 2
            'regime_name' — human-readable label: 'calm', 'stressed',
                            or 'crisis'
    """
    panel = panel.copy()
    # .copy() prevents pandas from modifying the original DataFrame
    # that was passed in — a good habit whenever you're adding columns

    # Start by assigning every row to regime 1 (stressed) as the default
    panel["regime"] = 1

    # Then overwrite with the correct label for calm and crisis months
    panel.loc[panel["VIX"] < VIX_CALM_THRESHOLD,   "regime"] = 0
    panel.loc[panel["VIX"] >= VIX_CRISIS_THRESHOLD, "regime"] = 2
    # .loc[] selects rows where the condition is True and assigns
    # the value only to those rows. Rows that match neither condition
    # keep the default value of 1 (stressed).

    # Add a human-readable name column alongside the integer label.
    # The integer label is what your models will use (cleaner for
    # indexing). The name column is for plots, tables, and the paper.
    regime_map = {0: "calm", 1: "stressed", 2: "crisis"}
    panel["regime_name"] = panel["regime"].map(regime_map)

    return panel


def print_regime_summary(panel_with_regimes: pd.DataFrame) -> None:
    """
    Prints a summary table showing how many months fall into each
    regime, and what percentage of the total dataset each represents.
    Also prints the specific months labeled as crisis, so you can
    visually verify they match known historical events.
    """
    total = len(panel_with_regimes)

    print("=" * 45)
    print("REGIME DISTRIBUTION SUMMARY")
    print("=" * 45)

    for regime_id, name in [(0, "calm"), (1, "stressed"), (2, "crisis")]:
        subset = panel_with_regimes[panel_with_regimes["regime"] == regime_id]
        count = len(subset)
        pct   = count / total * 100
        print(f"  {name.capitalize():10s} (regime {regime_id}): "
              f"{count:3d} months  ({pct:.1f}%)")

    print("=" * 45)

    # Print all crisis months explicitly — this is your key validation
    # step. October 2008 and March 2020 must appear in this list.
    crisis_months = panel_with_regimes[
        panel_with_regimes["regime"] == 2
    ][["VIX", "regime_name"]]

    print(f"\nAll {len(crisis_months)} crisis months (VIX >= {VIX_CRISIS_THRESHOLD}):")
    print(crisis_months.to_string())


def plot_regime_timeline(panel_with_regimes: pd.DataFrame,
                         output_path: str) -> None:
    """
    Creates the regime timeline visualization — a VIX line chart with
    colored background bands showing which regime each month belongs to.

    This becomes one of your key figures in the paper. It is the
    visual proof that your regime detection is working correctly and
    that the three regimes correspond to economically meaningful
    distinct periods.
    """
    fig, ax = plt.subplots(figsize=(14, 5))

    # Color palette for the three regimes.
    # Green for calm (things are fine), amber for stressed (watch out),
    # red for crisis (everything is on fire).
    colors = {0: "#d4edda", 1: "#fff3cd", 2: "#f8d7da"}
    # These are soft, muted tones so they don't overwhelm the VIX line
    # plotted on top.

    # Draw the colored background bands.
    # We iterate through the panel row by row and shade each month's
    # background according to its regime. We use axvspan(), which
    # draws a vertical shaded rectangle between two x-axis values.
    dates = panel_with_regimes.index
    for i in range(len(dates)):
        regime = panel_with_regimes["regime"].iloc[i]
        start  = dates[i]
        # The end of this month's band is the start of the next month's
        # band, or 3 months after the last date if this is the last row.
        end = dates[i + 1] if i < len(dates) - 1 else dates[i] + pd.DateOffset(months=3)
        ax.axvspan(start, end, color=colors[regime], alpha=0.6)

    # Plot the VIX line on top of the colored bands
    ax.plot(panel_with_regimes.index,
            panel_with_regimes["VIX"],
            color="black", linewidth=1.2, label="VIX", zorder=5)
    # zorder=5 ensures the line is drawn on top of the background bands

    # Draw horizontal lines at your two threshold values
    ax.axhline(VIX_CALM_THRESHOLD,
               color="#28a745", linestyle="--",
               linewidth=1, alpha=0.8, label=f"Calm threshold ({VIX_CALM_THRESHOLD})")
    ax.axhline(VIX_CRISIS_THRESHOLD,
               color="#dc3545", linestyle="--",
               linewidth=1, alpha=0.8, label=f"Crisis threshold ({VIX_CRISIS_THRESHOLD})")

    # Build a legend manually for the colored regime bands.
    # ax.legend() can't pick these up automatically because they
    # were drawn with axvspan(), not with labelled plot() calls.
    legend_patches = [
        mpatches.Patch(color=colors[0], alpha=0.6, label="Calm (VIX < 20)"),
        mpatches.Patch(color=colors[1], alpha=0.6, label="Stressed (20 ≤ VIX < 30)"),
        mpatches.Patch(color=colors[2], alpha=0.6, label="Crisis (VIX ≥ 30)"),
    ]
    ax.legend(handles=legend_patches + ax.get_legend_handles_labels()[0],
              loc="upper right", fontsize=9)

    ax.set_title("Market Regime Timeline (VIX-based, 1990–2024)",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("VIX Index Level")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.show()
    print(f"\nRegime timeline saved to {output_path}")


if __name__ == "__main__":
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("outputs/figures", exist_ok=True)

    # Load the raw-units panel — NOT the scaled one.
    # VIX thresholds are defined in real units; scaled values
    # would make the thresholds meaningless.
    raw_panel = pd.read_csv(
        "data/processed/macro_panel_raw_units.csv",
        index_col=0,
        parse_dates=True
    )

    # Assign regime labels
    panel_with_regimes = assign_regimes(raw_panel)

    # Print the summary and validate against known history
    print_regime_summary(panel_with_regimes)

    # Save regime labels alongside VIX for future reference
    panel_with_regimes[["VIX", "regime", "regime_name"]].to_csv(
        "data/processed/regime_labels.csv"
    )
    print("\nRegime labels saved to data/processed/regime_labels.csv")

    # Save the full panel with regimes attached for convenience
    panel_with_regimes.to_csv(
        "data/processed/macro_panel_with_regimes.csv"
    )

    # Generate the visualization
    plot_regime_timeline(
        panel_with_regimes,
        output_path="outputs/figures/03_regime_timeline.png"
    )