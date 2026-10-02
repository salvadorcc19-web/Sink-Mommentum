#!/usr/bin/env python3
"""
Reference implementation: minimal two-sink competitive dynamical model
======================================================================

Associated manuscript
---------------------
"A Dynamical Theory of Sink Establishment in Plants"

Purpose
-------
This script is the complete deterministic reference implementation used for
the conceptual demonstrations in Section 5. It generates Figures 2-4, Table 1,
the supplementary configuration table, and a diagnostics file.

The model is intentionally normalized and conceptual. It is NOT calibrated to a
specific crop, organ, developmental stage, environment, or dataset. Numerical
values and forcing functions are chosen only to illustrate the dynamical
structure of the proposed sink-momentum framework.

Core definitions
----------------
For each generic sink i in {1, 2}:

    Pi_i = M_i * v_i

where M_i is structural sink biomass/capacity, v_i is metabolic accumulation
velocity, and Pi_i is physiological sink momentum.

Dynamic sink strength is

    S_i = dPi_i/dt
        = M_i * dv_i/dt + v_i * dM_i/dt

so that the two additive contributions can be interpreted as:
    M_i * dv_i/dt  -> metabolic acceleration/deceleration
    v_i * dM_i/dt  -> structural reinforcement/loss

Metabolic accumulation velocity has its own state dynamics:

    dv_i/dt = (vhat_i(t) - v_i) / tau_v_i

The target vhat_i(t) is prescribed only to create two temporally distinct
developmental patterns. It is not fitted to empirical observations.

Resource and growth dynamics
----------------------------
The two sinks share a normalized carbon pool C:

    dC/dt = J_C(t) - U_1 - U_2 - k_C * C

with

    U_i = k_U * phi_i * v_i * g_C(C)
    g_C(C) = C / (K_C + C)
    dM_i/dt = Y_i * U_i

Relative allocation
-------------------
Allocation is obtained directly from the positive component of dynamic sink
strength, without a softmax or additional competition-response parameter:

    S_i^+ = max(S_i, 0)

If at least one sink has positive dynamic sink strength:

    phi_i,n+1 = S_i,n^+ / sum_j S_j,n^+

If all sinks satisfy S_i <= 0, the previous allocation is retained:

    phi_i,n+1 = phi_i,n

Because S_i contains dM_i/dt, and dM_i/dt depends on CURRENT allocation, the
model uses an explicit causal update:

    current phi_n
        -> current resource use
        -> current structural growth
        -> current S_n
        -> next allocation phi_(n+1)

This avoids a same-instant algebraic loop.

Numerical implementation
------------------------
- Deterministic explicit Euler integration
- Normalized time step DT = 0.02
- Normalized horizon T_END = 120
- No stochastic forcing
- No random initialization
- No fitted parameters
- No external data
- Non-negativity safeguard applied to C, M_i, and v_i after each update

Figure provenance
-----------------
Figure 2:
    Generated directly from the complete two-sink competitive simulation.

Figure 3:
    Generated from the SAME simulation as Figure 2. The code locates the first
    crossing Pi_1 = Pi_2 for which S_1 and S_2 have opposite signs and estimates
    the crossing time by linear interpolation.

Figure 4:
    Generated independently from idealized prescribed trajectories constructed
    specifically to illustrate the four canonical limiting regimes
    (growth-driven, activation-driven, mixed, weakening). It is not another
    realization of the competitive simulation.

Expected reference diagnostics
------------------------------
For the configuration below, the reference execution gives approximately:

    t0              = 57.76351 normalized time units
    Pi_1(t0)        = 2.34380152
    Pi_2(t0)        = 2.34380152
    S_1(t0)         = -0.16083507
    S_2(t0)         =  0.16501635

and the allocation simplex is conserved to floating-point precision.

Outputs
-------
Figure_2_minimal_competitive_dynamics.png/.pdf
Figure_3_equal_momentum_opposite_strength.png/.pdf
Figure_4_canonical_modes.png/.pdf
Table_1_canonical_regimes.csv/.png
Table_S1_minimal_model_configuration.csv/.png
minimal_model_diagnostics.txt

Dependencies
------------
Python 3.x
NumPy
Matplotlib

Run
---
    python sink_momentum_toy_model_reference_documented.py
"""

from pathlib import Path
import csv
import platform
import sys

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
# Output configuration
# ---------------------------------------------------------------------
OUT = Path("sink_momentum_toy_outputs_reference")
OUT.mkdir(parents=True, exist_ok=True)
SAVE_DPI = 600

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def sigmoid(x):
    """
    Numerically stable logistic function.

    The argument is clipped to [-60, 60] only to avoid floating-point overflow.
    This safeguard does not materially alter the trajectories in the range used
    by this conceptual example.
    """
    x = np.asarray(x, dtype=float)
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))


def positive_normalization(s, previous_phi, eps=1.0e-14):
    """
    Convert dynamic sink strength into allocation for the NEXT interval.

    S_i^+ = max(S_i, 0)

    If sum_j S_j^+ > eps:
        phi_i,n+1 = S_i,n^+ / sum_j S_j,n^+
    otherwise:
        phi_i,n+1 = phi_i,n

    No softmax, exponential competition transform, sensitivity coefficient, or
    allocation-response time constant is introduced.
    """
    s_pos = np.maximum(np.asarray(s, dtype=float), 0.0)
    total = float(np.sum(s_pos))
    if total > eps:
        return s_pos / total
    return np.asarray(previous_phi, dtype=float).copy()


# ---------------------------------------------------------------------
# Prescribed source supply and metabolic targets
# ---------------------------------------------------------------------
def source_carbon_input(t):
    """Prescribed normalized source-supply flux J_C(t); illustrative, not fitted."""
    t = np.asarray(t, dtype=float)
    return (
        0.35
        + 0.05 * np.sin(2.0 * np.pi * t / 30.0)
        + 0.08 * sigmoid((t - 20.0) / 8.0)
    )


def vhat_1(t):
    """Prescribed target for Sink 1: early activation followed by decline."""
    t = np.asarray(t, dtype=float)
    return (
        0.15
        + 1.30
        * sigmoid((t - 12.0) / 5.0)
        * (1.0 - sigmoid((t - 50.0) / 6.0))
    )


def vhat_2(t):
    """Prescribed target for Sink 2: later establishment and sustained activity."""
    t = np.asarray(t, dtype=float)
    return (
        0.08
        + 1.35
        * sigmoid((t - 42.0) / 6.0)
        * (1.0 - 0.15 * sigmoid((t - 100.0) / 10.0))
    )


# ---------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------
# All parameter values are normalized and illustrative; none are fitted.
P = {
    "K_C": 1.20,       # carbon half-saturation in g_C(C)
    "k_U": 0.17,       # metabolic-coordinate -> resource-use scaling
    "Y_1": 0.55,       # resource use -> structural growth, Sink 1
    "Y_2": 0.55,       # resource use -> structural growth, Sink 2
    "k_C": 0.04,       # first-order generic carbon loss [normalized time^-1]
    "tau_v_1": 5.0,    # metabolic response time, Sink 1
    "tau_v_2": 6.0,    # metabolic response time, Sink 2
}

DT = 0.02
T_END = 120.0
TIME = np.arange(0.0, T_END + DT, DT)

INITIAL = {
    "C": 4.0,
    "M_1": 1.20,
    "M_2": 0.80,
    "v_1": 0.10,
    "v_2": 0.05,
    "phi_1": 0.50,
    "phi_2": 0.50,
}


# ---------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------
def simulate():
    """
    Run the complete two-sink competitive simulation.

    Continuous states:
        C, M_1, M_2, v_1, v_2

    Sequential allocation:
        phi_1, phi_2

    Derived quantities:
        U_i, dM_i/dt, dv_i/dt, Pi_i, S_i

    Integration:
        explicit Euler, x_(n+1) = x_n + DT * dx/dt |_n

    Allocation for interval n+1 is computed from S_n. This ordering is
    deliberate and prevents a same-instant algebraic dependency between
    allocation and structural growth.
    """
    t = TIME
    n = len(t)

    C = np.zeros(n)
    M1 = np.zeros(n)
    M2 = np.zeros(n)
    v1 = np.zeros(n)
    v2 = np.zeros(n)
    phi1 = np.zeros(n)
    phi2 = np.zeros(n)

    U1 = np.zeros(n)
    U2 = np.zeros(n)
    dM1 = np.zeros(n)
    dM2 = np.zeros(n)
    dv1 = np.zeros(n)
    dv2 = np.zeros(n)
    S1 = np.zeros(n)
    S2 = np.zeros(n)

    C[0] = INITIAL["C"]
    M1[0] = INITIAL["M_1"]
    M2[0] = INITIAL["M_2"]
    v1[0] = INITIAL["v_1"]
    v2[0] = INITIAL["v_2"]
    phi1[0] = INITIAL["phi_1"]
    phi2[0] = INITIAL["phi_2"]

    for k in range(n - 1):
        # Current carbon availability.
        gC = C[k] / (P["K_C"] + C[k])

        # Current allocation acts over the present interval.
        U1[k] = P["k_U"] * phi1[k] * v1[k] * gC
        U2[k] = P["k_U"] * phi2[k] * v2[k] * gC

        # Structural growth.
        dM1[k] = P["Y_1"] * U1[k]
        dM2[k] = P["Y_2"] * U2[k]

        # Metabolic accumulation velocity evolves independently toward its target.
        dv1[k] = (float(vhat_1(t[k])) - v1[k]) / P["tau_v_1"]
        dv2[k] = (float(vhat_2(t[k])) - v2[k]) / P["tau_v_2"]

        # Dynamic sink strength from the product rule.
        S1[k] = M1[k] * dv1[k] + v1[k] * dM1[k]
        S2[k] = M2[k] * dv2[k] + v2[k] * dM2[k]

        # Direct relative normalization of positive dynamic sink strength.
        # This allocation is applied to the NEXT numerical interval.
        phi_next = positive_normalization(
            [S1[k], S2[k]],
            [phi1[k], phi2[k]],
        )

        # Shared carbon balance.
        dC = (
            float(source_carbon_input(t[k]))
            - U1[k]
            - U2[k]
            - P["k_C"] * C[k]
        )

        # Explicit state update.
        C[k + 1] = max(C[k] + DT * dC, 0.0)
        M1[k + 1] = max(M1[k] + DT * dM1[k], 0.0)
        M2[k + 1] = max(M2[k] + DT * dM2[k], 0.0)
        v1[k + 1] = max(v1[k] + DT * dv1[k], 0.0)
        v2[k + 1] = max(v2[k] + DT * dv2[k], 0.0)
        phi1[k + 1], phi2[k + 1] = phi_next

    # Last-point diagnostics.
    k = n - 1
    gC = C[k] / (P["K_C"] + C[k])

    U1[k] = P["k_U"] * phi1[k] * v1[k] * gC
    U2[k] = P["k_U"] * phi2[k] * v2[k] * gC
    dM1[k] = P["Y_1"] * U1[k]
    dM2[k] = P["Y_2"] * U2[k]
    dv1[k] = (float(vhat_1(t[k])) - v1[k]) / P["tau_v_1"]
    dv2[k] = (float(vhat_2(t[k])) - v2[k]) / P["tau_v_2"]
    S1[k] = M1[k] * dv1[k] + v1[k] * dM1[k]
    S2[k] = M2[k] * dv2[k] + v2[k] * dM2[k]

    Pi1 = M1 * v1
    Pi2 = M2 * v2

    result = {
        "t": t,
        "C": C,
        "M1": M1,
        "M2": M2,
        "v1": v1,
        "v2": v2,
        "vhat1": vhat_1(t),
        "vhat2": vhat_2(t),
        "phi1": phi1,
        "phi2": phi2,
        "U1": U1,
        "U2": U2,
        "dM1": dM1,
        "dM2": dM2,
        "dv1": dv1,
        "dv2": dv2,
        "S1": S1,
        "S2": S2,
        "Pi1": Pi1,
        "Pi2": Pi2,
    }

    validate_simulation(result)
    return result


# ---------------------------------------------------------------------
# Internal consistency checks
# ---------------------------------------------------------------------
def validate_simulation(r):
    """
    Check numerical consistency of the reference run.

    These are traceability/debugging checks, not biological validation tests.
    """
    finite_names = [
        "C", "M1", "M2", "v1", "v2",
        "phi1", "phi2", "Pi1", "Pi2", "S1", "S2",
    ]

    for name in finite_names:
        if not np.all(np.isfinite(r[name])):
            raise FloatingPointError(f"Non-finite values detected in {name}.")

    # C, M_i, and v_i are explicitly constrained to remain nonnegative.
    for name in ["C", "M1", "M2", "v1", "v2"]:
        if np.min(r[name]) < -1.0e-12:
            raise AssertionError(f"Negative state detected in {name}.")

    # Direct normalization should preserve the two-sink allocation simplex.
    simplex_error = np.max(np.abs(r["phi1"] + r["phi2"] - 1.0))
    if simplex_error > 1.0e-12:
        raise AssertionError(
            f"Allocation simplex error too large: {simplex_error:.3e}"
        )

    if min(np.min(r["phi1"]), np.min(r["phi2"])) < -1.0e-12:
        raise AssertionError("Negative allocation fraction detected.")


# ---------------------------------------------------------------------
# Equal-momentum crossing with opposite dynamic sink strength
# ---------------------------------------------------------------------
def find_equal_momentum_crossing(r):
    """
    Find the first Pi_1 = Pi_2 crossing with opposite signs of S_1 and S_2.

    The crossing time t0 is estimated by linear interpolation of
    delta(t) = Pi_1(t) - Pi_2(t). Pi_i and S_i are then interpolated to t0.

    Figure 3 is therefore extracted from the SAME trajectory used for Figure 2.
    """
    t = r["t"]
    delta = r["Pi1"] - r["Pi2"]
    candidates = np.where(delta[:-1] * delta[1:] <= 0.0)[0]

    for k in candidates:
        # Require opposite signs on the local interval.
        if r["S1"][k] * r["S2"][k] < 0.0:
            x1, x2 = t[k], t[k + 1]
            y1, y2 = delta[k], delta[k + 1]

            if abs(y2 - y1) < 1.0e-15:
                t0 = x1
            else:
                t0 = x1 - y1 * (x2 - x1) / (y2 - y1)

            crossing = {
                "t0": t0,
                "Pi1": np.interp(t0, t, r["Pi1"]),
                "Pi2": np.interp(t0, t, r["Pi2"]),
                "S1": np.interp(t0, t, r["S1"]),
                "S2": np.interp(t0, t, r["S2"]),
            }

            if abs(crossing["Pi1"] - crossing["Pi2"]) > 1.0e-8:
                raise AssertionError("Inconsistent equal-momentum interpolation.")

            if crossing["S1"] * crossing["S2"] >= 0.0:
                raise AssertionError(
                    "Crossing does not have opposite dynamic-strength signs."
                )

            return crossing

    raise RuntimeError(
        "No equal-momentum crossing with opposite dynamic-strength signs was found."
    )


# ---------------------------------------------------------------------
# Figure 2: complete minimal competitive dynamics
# ---------------------------------------------------------------------
def make_figure_2(r):
    """
    Generate Figure 2 directly from the complete competitive simulation.

    Panels: C; vhat_i and v_i; M_i; Pi_i; S_i; phi_i.
    """
    t = r["t"]
    fig, axes = plt.subplots(3, 2, figsize=(11.5, 12.0), constrained_layout=True)
    ax = axes.ravel()

    ax[0].plot(t, r["C"])
    ax[0].set_title("A) Shared carbon resource pool")
    ax[0].set_ylabel(r"Shared carbon pool, $C$")

    # Use one default color per sink and reuse it for target/state curves.
    line_v1, = ax[1].plot(t, r["v1"], label=r"Sink 1 state, $v_1$")
    ax[1].plot(
        t, r["vhat1"], "--",
        color=line_v1.get_color(),
        label=r"Sink 1 target, $\hat{v}_1$"
    )
    line_v2, = ax[1].plot(t, r["v2"], label=r"Sink 2 state, $v_2$")
    ax[1].plot(
        t, r["vhat2"], "--",
        color=line_v2.get_color(),
        label=r"Sink 2 target, $\hat{v}_2$"
    )
    ax[1].set_title("B) Metabolic accumulation velocity")
    ax[1].set_ylabel(r"Metabolic accumulation velocity, $v_i$")
    ax[1].legend(frameon=False, ncol=2)

    ax[2].plot(t, r["M1"], label="Sink 1")
    ax[2].plot(t, r["M2"], label="Sink 2")
    ax[2].set_title("C) Structural biomass")
    ax[2].set_ylabel(r"Structural biomass, $M_i$")
    ax[2].legend(frameon=False)

    ax[3].plot(t, r["Pi1"], label=r"Sink 1, $\Pi_1$")
    ax[3].plot(t, r["Pi2"], label=r"Sink 2, $\Pi_2$")
    ax[3].set_title("D) Physiological sink momentum")
    ax[3].set_ylabel(r"Physiological sink momentum, $\Pi_i=M_i v_i$")
    ax[3].legend(frameon=False)

    ax[4].plot(t, r["S1"], label=r"Sink 1, $S_1$")
    ax[4].plot(t, r["S2"], label=r"Sink 2, $S_2$")
    ax[4].set_title("E) Dynamic sink strength")
    ax[4].set_ylabel(r"Dynamic sink strength, $S_i=d\Pi_i/dt$")
    ax[4].legend(frameon=False)

    ax[5].plot(t, r["phi1"], label=r"Sink 1, $\phi_1$")
    ax[5].plot(t, r["phi2"], label=r"Sink 2, $\phi_2$")
    ax[5].set_title("F) Relative allocation from dynamic sink strength")
    ax[5].set_ylabel(r"Relative allocation, $\phi_i$")
    ax[5].set_ylim(-0.02, 1.02)
    ax[5].legend(frameon=False)

    for a in ax:
        a.set_xlabel("Developmental time [normalized time units]")
        a.grid(alpha=0.20)

    fig.savefig(
        OUT / "Figure_2_minimal_competitive_dynamics.png",
        dpi=SAVE_DPI,
        bbox_inches="tight",
    )
    fig.savefig(
        OUT / "Figure_2_minimal_competitive_dynamics.pdf",
        bbox_inches="tight",
    )
    plt.close(fig)


# ---------------------------------------------------------------------
# Figure 3: equal momentum at t0, opposite dynamic strength at t0
# ---------------------------------------------------------------------
def make_figure_3(r, crossing):
    """
    Generate Figure 3 from the same trajectory used for Figure 2.

    The plotted window is centered on the interpolated time t0 at which
    Pi_1(t0) = Pi_2(t0) while S_1 and S_2 have opposite signs.
    """
    t0 = crossing["t0"]
    t = r["t"]
    window = 18.0
    mask = (t >= t0 - window) & (t <= t0 + window)

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.5), constrained_layout=True)

    p1, = axes[0].plot(t[mask], r["Pi1"][mask], label="Sink 1")
    p2, = axes[0].plot(t[mask], r["Pi2"][mask], label="Sink 2")

    # The crossing is identified by the t0 annotation.
    # No overlapping marker is added here because both sinks occupy the same point.
    axes[0].annotate(
        r"$t_0$",
        xy=(t0, crossing["Pi1"]),
        xytext=(t0 + 2.0, crossing["Pi1"] + 0.65),
        arrowprops={"arrowstyle": "->"},
    )
    axes[0].set_title(r"A) Equal sink momentum at $t_0$")
    axes[0].set_xlabel("Developmental time [normalized time units]")
    axes[0].set_ylabel(r"Physiological sink momentum, $\Pi_i=M_i v_i$")
    axes[0].legend(frameon=False)
    axes[0].grid(alpha=0.20)

    s1_line, = axes[1].plot(t[mask], r["S1"][mask], label="Sink 1")
    s2_line, = axes[1].plot(t[mask], r["S2"][mask], label="Sink 2")

    axes[1].plot(
        t0, crossing["S1"], "o",
        color=s1_line.get_color(),
        markersize=6,
    )
    axes[1].plot(
        t0, crossing["S2"], "o",
        color=s2_line.get_color(),
        markersize=6,
    )
    axes[1].annotate(
        r"$t_0$",
        xy=(t0, 0.0),
        xytext=(t0 + 2.0, 0.055),
        arrowprops={"arrowstyle": "->"},
    )
    axes[1].set_title(r"B) Opposite dynamic sink strength at $t_0$")
    axes[1].set_xlabel("Developmental time [normalized time units]")
    axes[1].set_ylabel(r"Dynamic sink strength, $S_i=d\Pi_i/dt$")
    axes[1].legend(frameon=False)
    axes[1].grid(alpha=0.20)

    fig.savefig(
        OUT / "Figure_3_equal_momentum_opposite_strength.png",
        dpi=SAVE_DPI,
        bbox_inches="tight",
    )
    fig.savefig(
        OUT / "Figure_3_equal_momentum_opposite_strength.pdf",
        bbox_inches="tight",
    )
    plt.close(fig)


# ---------------------------------------------------------------------
# Figure 4: canonical dynamical regimes
# ---------------------------------------------------------------------
def logistic_curve(t, low, high, midpoint, width):
    """Increasing logistic trajectory used only for the Figure 4 archetypes."""
    return low + (high - low) * sigmoid((t - midpoint) / width)


def decreasing_logistic(t, high, low, midpoint, width):
    """Decreasing logistic trajectory used only for the Figure 4 archetypes."""
    return low + (high - low) * (1.0 - sigmoid((t - midpoint) / width))


def make_figure_4():
    """
    Generate Figure 4 from independent idealized trajectories.

    IMPORTANT:
    Figure 4 is NOT another output of the competitive two-sink simulation.
    It isolates four limiting regimes implied by the decomposition of S_i.

    The plotted weakening example has both structural and metabolic decline.
    More generally, S_i < 0 can also occur when only one contribution is
    negative and its magnitude exceeds the positive contribution of the other.
    """
    tc = np.linspace(0.0, 100.0, 1001)

    regimes = [
        (
            "Growth-driven",
            logistic_curve(tc, 1.0, 5.0, 48.0, 9.0),
            np.full_like(tc, 1.0),
        ),
        (
            "Activation-driven",
            np.full_like(tc, 3.0),
            logistic_curve(tc, 0.25, 1.25, 48.0, 8.0),
        ),
        (
            "Mixed",
            logistic_curve(tc, 1.0, 5.0, 46.0, 10.0),
            logistic_curve(tc, 0.25, 1.20, 52.0, 9.0),
        ),
        (
            "Weakening",
            decreasing_logistic(tc, 4.0, 3.2, 58.0, 12.0),
            decreasing_logistic(tc, 1.15, 0.25, 50.0, 8.0),
        ),
    ]

    fig, axes = plt.subplots(
        4, 3,
        figsize=(11.8, 12.8),
        sharex=True,
        constrained_layout=True,
    )

    for row, (name, M, v) in enumerate(regimes):
        dM = np.gradient(M, tc)
        dv = np.gradient(v, tc)

        structural = v * dM
        metabolic = M * dv
        S = structural + metabolic

        axes[row, 0].plot(tc, M)
        axes[row, 1].plot(tc, v)

        s_line, = axes[row, 2].plot(tc, S, label=r"$S_i$")
        axes[row, 2].plot(
            tc, structural, "--",
            label=r"$v_i\,dM_i/dt$",
        )
        axes[row, 2].plot(
            tc, metabolic, ":",
            label=r"$M_i\,dv_i/dt$",
        )

        axes[row, 0].set_ylabel(name)
        for col in range(3):
            axes[row, col].grid(alpha=0.20)

    axes[0, 0].set_title(r"A) Structural biomass, $M_i$")
    axes[0, 1].set_title(r"B) Metabolic accumulation velocity, $v_i$")
    axes[0, 2].set_title(r"C) Dynamic sink strength and components")
    axes[0, 2].legend(frameon=False)

    for col in range(3):
        axes[-1, col].set_xlabel("Developmental time [normalized time units]")

    fig.savefig(
        OUT / "Figure_4_canonical_modes.png",
        dpi=SAVE_DPI,
        bbox_inches="tight",
    )
    fig.savefig(
        OUT / "Figure_4_canonical_modes.pdf",
        bbox_inches="tight",
    )
    plt.close(fig)


# ---------------------------------------------------------------------
# Table 1
# ---------------------------------------------------------------------
def make_table_1():
    """Export the canonical-regime table as CSV and a PNG review preview."""
    rows = [
        [
            "Growth-driven",
            "v_i dM_i/dt >> M_i dv_i/dt",
            "Sink strengthening is dominated by increasing structural capacity.",
            "Leaf expansion; stem elongation; root proliferation; fruit growth during cell expansion.",
        ],
        [
            "Activation-driven",
            "M_i dv_i/dt >> v_i dM_i/dt",
            "Sink strengthening is dominated by increasing metabolic competence at approximately unchanged structural size.",
            "Grain-filling initiation; tuber induction; activation of storage metabolism; hormonal activation.",
        ],
        [
            "Mixed",
            "M_i dv_i/dt ~ v_i dM_i/dt",
            "Structural reinforcement and metabolic acceleration contribute at comparable magnitude.",
            "Developing fruits; expanding leaves; storage organs; rapidly growing meristems.",
        ],
        [
            "Weakening",
            "S_i < 0",
            "Physiological sink momentum decreases because metabolic deceleration, structural loss, or their combined effect produces S_i < 0.",
            "Developmental completion; sink exhaustion; metabolic down-regulation; senescence.",
        ],
    ]

    headers = [
        "Canonical regime",
        "Dominant condition",
        "Dynamical interpretation",
        "Representative examples",
    ]

    csv_path = OUT / "Table_1_canonical_regimes.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)

    fig, ax = plt.subplots(figsize=(14.0, 4.6))
    ax.axis("off")
    table = ax.table(
        cellText=rows,
        colLabels=headers,
        loc="center",
        cellLoc="left",
        colLoc="left",
        colWidths=[0.14, 0.20, 0.31, 0.35],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1.0, 2.2)
    ax.set_title(
        "Table 1. Canonical regimes of sink establishment and decline",
        pad=12,
    )
    fig.savefig(
        OUT / "Table_1_canonical_regimes.png",
        dpi=SAVE_DPI,
        bbox_inches="tight",
    )
    plt.close(fig)


# ---------------------------------------------------------------------
# Supplementary configuration table
# ---------------------------------------------------------------------
def make_configuration_table():
    """
    Export all numerical values and update rules needed to reproduce the toy model.

    The CSV is the machine-readable record; the PNG is only a visual preview.
    """
    rows = [
        ["Simulation horizon", "T", f"{T_END:g}", "normalized time units"],
        ["Numerical step", "Delta t", f"{DT:g}", "normalized time units"],
        ["Initial shared carbon pool", "C(0)", f"{INITIAL['C']:g}", "normalized carbon-pool units"],
        ["Initial structural biomass, sink 1", "M_1(0)", f"{INITIAL['M_1']:g}", "normalized structural units"],
        ["Initial structural biomass, sink 2", "M_2(0)", f"{INITIAL['M_2']:g}", "normalized structural units"],
        ["Initial metabolic velocity, sink 1", "v_1(0)", f"{INITIAL['v_1']:g}", "normalized metabolic-velocity units"],
        ["Initial metabolic velocity, sink 2", "v_2(0)", f"{INITIAL['v_2']:g}", "normalized metabolic-velocity units"],
        ["Initial allocation, sink 1", "phi_1(0)", f"{INITIAL['phi_1']:g}", "fraction"],
        ["Initial allocation, sink 2", "phi_2(0)", f"{INITIAL['phi_2']:g}", "fraction"],
        ["Carbon half-saturation", "K_C", f"{P['K_C']:g}", "normalized carbon-pool units"],
        ["Resource-use scaling", "k_U", f"{P['k_U']:g}", "normalized conversion scale"],
        ["Growth conversion, sink 1", "Y_1", f"{P['Y_1']:g}", "normalized conversion coefficient"],
        ["Growth conversion, sink 2", "Y_2", f"{P['Y_2']:g}", "normalized conversion coefficient"],
        ["Generic carbon loss", "k_C", f"{P['k_C']:g}", "normalized time^-1"],
        ["Metabolic response time, sink 1", "tau_v,1", f"{P['tau_v_1']:g}", "normalized time units"],
        ["Metabolic response time, sink 2", "tau_v,2", f"{P['tau_v_2']:g}", "normalized time units"],
        ["Partition rule", "phi_i,n+1", "S_i,n^+ / sum_j S_j,n^+", "direct relative normalization"],
        ["All S_i <= 0 fallback", "phi_i,n+1", "phi_i,n", "previous allocation retained"],
    ]

    headers = ["Quantity", "Symbol", "Value / rule", "Units / interpretation"]

    csv_path = OUT / "Table_S1_minimal_model_configuration.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)

    fig, ax = plt.subplots(figsize=(11.5, 8.8))
    ax.axis("off")
    table = ax.table(
        cellText=rows,
        colLabels=headers,
        loc="center",
        cellLoc="left",
        colLoc="left",
        colWidths=[0.36, 0.16, 0.23, 0.25],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.0)
    table.scale(1.0, 1.45)
    ax.set_title(
        "Supplementary Table S1. Minimal competitive dynamical model configuration",
        pad=12,
    )
    fig.savefig(
        OUT / "Table_S1_minimal_model_configuration.png",
        dpi=SAVE_DPI,
        bbox_inches="tight",
    )
    plt.close(fig)


# ---------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------
def write_diagnostics(r, crossing):
    """
    Write numerical and software traceability information.

    These diagnostics are not additional biological results. They are reference
    values that allow an independent user to verify that the exact script and
    numerical environment reproduce the published conceptual demonstration.
    """
    max_partition_error = np.max(
        np.abs(r["phi1"] + r["phi2"] - 1.0)
    )

    text = f"""Minimal two-sink model reproducibility diagnostics

Software environment
--------------------
Python:      {sys.version.replace(chr(10), ' ')}
NumPy:       {np.__version__}
Matplotlib:  {matplotlib.__version__}
Platform:    {platform.platform()}

Numerical configuration
-----------------------
Integrator:  explicit Euler
DT:          {DT}
T_END:       {T_END}
Time points: {len(TIME)}
Stochastic operations: none
Fitted parameters: none
External data: none

Equal-momentum crossing
-----------------------
t0 = {crossing['t0']:.8f} normalized time units
Pi_1(t0) = {crossing['Pi1']:.10f}
Pi_2(t0) = {crossing['Pi2']:.10f}
S_1(t0) = {crossing['S1']:.10f}
S_2(t0) = {crossing['S2']:.10f}

Local interpretation at t0
--------------------------
Sink 1: {'strengthening' if crossing['S1'] > 0 else 'weakening'}
Sink 2: {'strengthening' if crossing['S2'] > 0 else 'weakening'}

Allocation checks
-----------------
max |phi_1 + phi_2 - 1| = {max_partition_error:.3e}
min phi_1 = {np.min(r['phi1']):.10f}
min phi_2 = {np.min(r['phi2']):.10f}
max phi_1 = {np.max(r['phi1']):.10f}
max phi_2 = {np.max(r['phi2']):.10f}

State ranges
------------
C:   [{np.min(r['C']):.8f}, {np.max(r['C']):.8f}]
M_1: [{np.min(r['M1']):.8f}, {np.max(r['M1']):.8f}]
M_2: [{np.min(r['M2']):.8f}, {np.max(r['M2']):.8f}]
v_1: [{np.min(r['v1']):.8f}, {np.max(r['v1']):.8f}]
v_2: [{np.min(r['v2']):.8f}, {np.max(r['v2']):.8f}]

Figure provenance
-----------------
Figures 2 and 3: same competitive two-sink simulation.
Figure 4: independent idealized canonical trajectories.

Scope
-----
All quantities are normalized and conceptual.
The model is not calibrated to a particular crop or dataset.
"""

    diagnostic_path = OUT / "minimal_model_diagnostics.txt"
    diagnostic_path.write_text(text, encoding="utf-8")
    print(text)


def main():
    """
    Execute the full reproducibility workflow and regenerate all outputs.
    """
    r = simulate()
    crossing = find_equal_momentum_crossing(r)
    make_figure_2(r)
    make_figure_3(r, crossing)
    make_figure_4()
    make_table_1()
    make_configuration_table()
    write_diagnostics(r, crossing)
    print(f"Outputs written to: {OUT.resolve()}")


if __name__ == "__main__":
    main()
