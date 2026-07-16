import os
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# Sink momentum theoretical mini-model
#
# M_i(t)  = sink structural biomass / capacity [gC]
# v_i(t)  = sink flow velocity in metabolic/flux space [mol s^-1]
#
# Pi_i(t) = M_i(t) v_i(t)
#         = physiological sink momentum [gC mol s^-1]
#
# S_i(t)  = dPi_i/dt
#         = dynamic sink strength / sink-force analogue [gC mol s^-2]
#
# dPi_i/dt = M_i dv_i/dt + v_i dM_i/dt
# ============================================================


def sigmoid(x, k=1.0):
    return 1.0 / (1.0 + np.exp(-k * x))


def rk4_step(f, y, t, dt, params):
    k1 = f(t, y, params)
    k2 = f(t + 0.5 * dt, y + 0.5 * dt * k1, params)
    k3 = f(t + 0.5 * dt, y + 0.5 * dt * k2, params)
    k4 = f(t + dt, y + dt * k3, params)
    return y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def source_supply(t, params):
    A0 = params["A0"]
    amp = params["A_amp"]
    period = params["period"]
    return A0 * (1.0 + amp * np.sin(2.0 * np.pi * t / period))


def sink_ode(t, y, p):
    """
    State variables:
    C = soluble carbon pool [gC]
    M = sink structural biomass / active sink size [gC]
    v = sink flow velocity in metabolic/flux space [mol s^-1]

    Pi = M*v is diagnosed as sink momentum.
    dPi/dt is diagnosed as dynamic sink strength.
    """

    C, M, v = y

    A = source_supply(t, p)

    # Carbon availability gate
    gC = C / (p["K_C"] + C)

    # Developmental/metabolic activation signal
    dev = sigmoid(t - p["t_dev"], k=p["k_dev"])

    # Target flow velocity in metabolic/flux space
    v_target = p["v_base"] + p["v_max"] * dev * gC

    # Sink flow-velocity kinetics
    dvdt = (v_target - v) / p["tau_v"]

    # Physiological sink momentum
    Pi = M * v

    # Biomass accumulation driven by sink momentum and carbon availability
    dMdt = p["Yg"] * Pi * gC

    # Carbon pool balance
    dCdt = A - Pi * gC - p["k_loss"] * C

    return np.array([dCdt, dMdt, dvdt])


def simulate(params):
    t0, tf, dt = params["t0"], params["tf"], params["dt"]
    t = np.arange(t0, tf + dt, dt)

    y = np.zeros((len(t), 3))
    y[0, :] = [params["C0"], params["M0"], params["v0"]]

    for i in range(len(t) - 1):
        y[i + 1, :] = rk4_step(sink_ode, y[i, :], t[i], dt, params)
        y[i + 1, :] = np.maximum(y[i + 1, :], 0.0)

    C = y[:, 0]
    M = y[:, 1]
    v = y[:, 2]

    dydt = np.array([sink_ode(ti, yi, params) for ti, yi in zip(t, y)])
    dCdt = dydt[:, 0]
    dMdt = dydt[:, 1]
    dvdt = dydt[:, 2]

    Pi = M * v

    structural_term = v * dMdt
    activation_term = M * dvdt
    dPi_dt = structural_term + activation_term

    return {
        "t": t,
        "C": C,
        "M": M,
        "v": v,
        "dCdt": dCdt,
        "dMdt": dMdt,
        "dvdt": dvdt,
        "Pi": Pi,
        "dPi_dt": dPi_dt,
        "structural_term": structural_term,
        "activation_term": activation_term,
    }


def style_axis(ax, title=None):
    if title:
        ax.set_title(title, loc="left", fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.25)


def save_single_line_panel(t, y, ylabel, title, filename, xlabel="Time [s]", hline=False):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(t, y, linewidth=2)
    if hline:
        ax.axhline(0, linewidth=0.8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    style_axis(ax, title)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()


# ============================================================
# Figure 2: Full ODE trajectory
# ============================================================

def plot_dynamic_system(res, outdir="figures"):
    os.makedirs(outdir, exist_ok=True)
    t = res["t"]

    fig, axes = plt.subplots(3, 2, figsize=(13, 9), sharex=True)

    axes = axes.flatten()


    # -------------------------------------------------------
    # Left column
    # -------------------------------------------------------

    axes[0].plot(t, res["C"], linewidth=2)
    axes[0].set_ylabel("C pool\n[gC]")
    style_axis(axes[0], "A. Carbon availability")

    axes[2].plot(t, res["v"], linewidth=2)
    axes[2].set_ylabel("v\n[mol s$^{-1}$]")
    style_axis(axes[2], "B. Sink flow velocity")

    axes[4].plot(t, res["M"], linewidth=2)
    axes[4].set_ylabel("M\n[gC]")
    axes[4].set_xlabel("Time [s]")
    style_axis(axes[4], "C. Sink structural biomass")


    # -------------------------------------------------------
    # Right column
    # -------------------------------------------------------

    axes[1].plot(t, res["Pi"], linewidth=2)
    axes[1].set_ylabel(r"$\Pi=Mv$" "\n[gC mol s$^{-1}$]")
    style_axis(axes[1], "D. Sink momentum")

    axes[3].plot(t, res["dPi_dt"], linewidth=2)
    axes[3].axhline(0, linewidth=0.8)
    axes[3].set_ylabel(r"$d\Pi/dt$" "\n[gC mol s$^{-2}$]")
    style_axis(axes[3], "E. Dynamic sink strength")

    axes[5].plot(
        t,
        res["dPi_dt"],
        linewidth=2,
        label=r"Net: $d\Pi/dt$"
    )

    axes[5].plot(
        t,
        res["structural_term"],
        "--",
        linewidth=2,
        label=r"Structural term: $v\,dM/dt$"
    )

    axes[5].plot(
        t,
        res["activation_term"],
        "--",
        linewidth=2,
        label=r"Metabolic acceleration term: $M\,dv/dt$"
    )

    axes[5].axhline(0, linewidth=0.8)

    axes[5].set_ylabel(
        "Terms\n[gC mol s$^{-2}$]"
    )

    axes[5].set_xlabel("Time [s]")

    axes[5].legend(
        frameon=False,
        fontsize=10,
        loc="best"
    )

    style_axis(
        axes[5],
        "F. Dynamic sink strength decomposition"
    )


    plt.tight_layout()

    plt.subplots_adjust(
        wspace=0.28,
        hspace=0.32
    )

    filename = os.path.join(
        outdir,
        "Figure2_dynamic_sink_ode_system.png"
    )

    plt.savefig(
        filename,
        dpi=600,
        bbox_inches="tight"
    )

    plt.close()

    save_single_line_panel(
        t, res["C"],
        "C pool [gC]",
        "A. Carbon availability",
        os.path.join(outdir, "Figure2A_carbon_availability.png"),
    )

    save_single_line_panel(
        t, res["v"],
        r"v [mol s$^{-1}$]",
        "B. Sink flow velocity",
        os.path.join(outdir, "Figure2B_sink_flow_velocity.png"),
    )

    save_single_line_panel(
        t, res["M"],
        "M [gC]",
        "C. Sink structural biomass",
        os.path.join(outdir, "Figure2C_sink_structural_biomass.png"),
    )

    save_single_line_panel(
        t, res["Pi"],
        r"$\Pi=Mv$ [gC mol s$^{-1}$]",
        "D. Sink momentum",
        os.path.join(outdir, "Figure2D_sink_momentum.png"),
    )

    save_single_line_panel(
        t, res["dPi_dt"],
        r"$d\Pi/dt$ [gC mol s$^{-2}$]",
        "E. Dynamic sink strength",
        os.path.join(outdir, "Figure2E_dynamic_sink_strength.png"),
        hline=True,
    )

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(t, res["dPi_dt"], linewidth=2, label=r"Net: $d\Pi/dt$")
    ax.plot(t, res["structural_term"], "--", linewidth=2, label=r"Structural: $v\,dM/dt$")
    ax.plot(t, res["activation_term"], "--", linewidth=2, label=r"Flow acceleration: $M\,dv/dt$")
    ax.axhline(0, linewidth=0.8)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel(r"Terms [gC mol s$^{-2}$]")
    ax.legend(frameon=False)
    style_axis(ax, "F. Decomposition of dynamic sink strength")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "Figure2F_dynamic_strength_decomposition.png"), dpi=300)
    plt.close()

    return filename


# ============================================================
# Figure 3: May-style conceptual counterexample
# ============================================================

def plot_counterexample(outdir="figures"):
    os.makedirs(outdir, exist_ok=True)

    t = np.linspace(0, 100, 1001)
    t0 = 50.0

    Pi_A = 1.0 + 0.01 * (t - t0)
    Pi_B = 1.0 - 0.01 * (t - t0)

    dPi_A = np.gradient(Pi_A, t)
    dPi_B = np.gradient(Pi_B, t)

    fig, axes = plt.subplots(2, 1, figsize=(7, 6), sharex=True)

    axes[0].plot(t, Pi_A, linewidth=2, label="Sink A")
    axes[0].plot(t, Pi_B, linewidth=2, label="Sink B")
    axes[0].axvline(t0, linestyle="--", linewidth=1)
    axes[0].scatter([t0, t0], [1.0, 1.0], zorder=5)
    axes[0].set_ylabel(r"$\Pi$ [gC mol s$^{-1}$]")
    axes[0].legend(frameon=False)
    style_axis(axes[0], "A. Equal sink momentum")

    axes[1].plot(t, dPi_A, linewidth=2, label=r"Sink A: $d\Pi/dt>0$")
    axes[1].plot(t, dPi_B, linewidth=2, label=r"Sink B: $d\Pi/dt<0$")
    axes[1].axhline(0, linewidth=0.8)
    axes[1].axvline(t0, linestyle="--", linewidth=1)
    axes[1].set_ylabel(r"$d\Pi/dt$ [gC mol s$^{-2}$]")
    axes[1].set_xlabel("Time [s]")
    axes[1].legend(frameon=False)
    style_axis(axes[1], "B. Opposite dynamic sink strength")

    plt.tight_layout()
    filename = os.path.join(outdir, "Figure3_equal_momentum_opposite_strength.png")
    plt.savefig(filename, dpi=600)
    plt.close()

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter([1, 1], [0.10, -0.10], s=80)
    ax.axhline(0, linewidth=0.8)
    ax.text(1.02, 0.10, "Sink A: strengthening", va="center")
    ax.text(1.02, -0.10, "Sink B: weakening", va="center")
    ax.set_xlabel(r"Sink momentum, $\Pi$ [gC mol s$^{-1}$]")
    ax.set_ylabel(r"Dynamic sink strength, $d\Pi/dt$ [gC mol s$^{-2}$]")
    ax.set_xlim(0.85, 1.35)
    ax.set_ylim(-0.15, 0.15)
    style_axis(ax, "C. Same momentum, opposite strength")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "Figure3C_momentum_strength_phase_plane.png"), dpi=300)
    plt.close()

    return filename


# ============================================================
# Figure 4: Mechanistic scenarios
# ============================================================

def compute_from_prescribed(t, M, v):
    dMdt = np.gradient(M, t)
    dvdt = np.gradient(v, t)
    Pi = M * v
    structural_term = v * dMdt
    activation_term = M * dvdt
    dPi_dt = structural_term + activation_term

    return {
        "M": M,
        "v": v,
        "Pi": Pi,
        "dMdt": dMdt,
        "dvdt": dvdt,
        "structural_term": structural_term,
        "activation_term": activation_term,
        "dPi_dt": dPi_dt,
    }


def make_conceptual_scenarios(t):
    scenarios = {}

    M1 = 5.0 + 95.0 * sigmoid(t - 55, k=0.08)
    v1 = np.full_like(t, 0.035)
    scenarios["Growth-driven sink"] = compute_from_prescribed(t, M1, v1)

    M2 = np.full_like(t, 45.0)
    v2 = 0.005 + 0.060 * sigmoid(t - 50, k=0.12)
    scenarios["Flow-activation-driven sink"] = compute_from_prescribed(t, M2, v2)

    M3 = 5.0 + 90.0 * sigmoid(t - 65, k=0.08)
    v3 = 0.005 + 0.055 * sigmoid(t - 40, k=0.12)
    scenarios["Mixed sink emergence"] = compute_from_prescribed(t, M3, v3)

    M4 = 80.0 - 5.0 * sigmoid(t - 65, k=0.05)
    v4 = 0.065 * (1.0 - sigmoid(t - 55, k=0.12)) + 0.006
    scenarios["Sink weakening"] = compute_from_prescribed(t, M4, v4)

    return scenarios


def plot_scenarios(outdir="figures"):
    os.makedirs(outdir, exist_ok=True)

    t = np.linspace(0, 120, 1201)
    scenarios = make_conceptual_scenarios(t)

    fig, axes = plt.subplots(4, 3, figsize=(12, 10), sharex=True)

    for row, (name, s) in enumerate(scenarios.items()):
        axes[row, 0].plot(t, s["M"], linewidth=2)
        axes[row, 0].set_ylabel(name + "\nM [gC]")
        style_axis(axes[row, 0])

        axes[row, 1].plot(t, s["v"], linewidth=2)
        axes[row, 1].set_ylabel(r"v [mol s$^{-1}$]")
        style_axis(axes[row, 1])

        axes[row, 2].plot(t, s["dPi_dt"], linewidth=2, label=r"$d\Pi/dt$")
        axes[row, 2].plot(t, s["structural_term"], "--", linewidth=1.8, label=r"$v\,dM/dt$")
        axes[row, 2].plot(t, s["activation_term"], "--", linewidth=1.8, label=r"$M\,dv/dt$")
        axes[row, 2].axhline(0, linewidth=0.8)
        axes[row, 2].set_ylabel(r"$d\Pi/dt$" "\n[gC mol s$^{-2}$]")
        style_axis(axes[row, 2])

    axes[0, 0].set_title("A. Structural biomass", fontweight="bold")
    axes[0, 1].set_title("B. Flow velocity", fontweight="bold")
    axes[0, 2].set_title("C. Dynamic sink strength", fontweight="bold")

    for ax in axes[-1, :]:
        ax.set_xlabel("Time [s]")

    axes[0, 2].legend(frameon=False)

    plt.tight_layout()
    filename = os.path.join(outdir, "Figure4_canonical_sink_modes.png")
    plt.savefig(filename, dpi=600)
    plt.close()

    return filename


# ============================================================
# Supplementary Figure S1: Mechanical analogy
# ============================================================

def plot_mechanical_analogy(outdir="figures"):
    os.makedirs(outdir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 5))

    left_x = 0.25
    right_x = 0.75

    mechanics = [
        "Mass (m)",
        "Velocity (u)",
        "Momentum (p = mu)",
        "Force (dp/dt)"
    ]

    physiology = [
        "Sink structural biomass (M)",
        "Flow velocity in metabolic space (v)",
        "Sink momentum (Π = Mv)",
        "Dynamic sink strength (dΠ/dt)"
    ]

    y = [0.8, 0.6, 0.4, 0.2]

    for yi, m, p in zip(y, mechanics, physiology):
        ax.text(left_x, yi, m, fontsize=12, ha="center", va="center",
                bbox=dict(boxstyle="round", fc="white"))
        ax.text(right_x, yi, p, fontsize=12, ha="center", va="center",
                bbox=dict(boxstyle="round", fc="white"))
        ax.annotate(
            "",
            xy=(right_x - 0.13, yi),
            xytext=(left_x + 0.13, yi),
            arrowprops=dict(arrowstyle="->", lw=2),
        )

    ax.text(0.25, 0.93, "Classical mechanics", ha="center", fontsize=13, fontweight="bold")
    ax.text(0.75, 0.93, "Sink dynamics analogy", ha="center", fontsize=13, fontweight="bold")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    plt.tight_layout()
    filename = os.path.join(outdir, "FigureS1_mechanical_analogy.png")
    plt.savefig(filename, dpi=300)
    plt.close()

    return filename


# ============================================================
# Supplementary Figure S2: Hydraulic analogy
# ============================================================

def plot_hydraulic_analogy(outdir="figures"):
    os.makedirs(outdir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(11, 5))

    labels = [
        "Source\navailability",
        "Carbon\npool",
        "Flow velocity\nv(t)",
        "Sink momentum\nΠ=Mv",
        "Dynamic strength\ndΠ/dt",
        "Growth\naccumulation"
    ]

    x = np.arange(len(labels))

    for xi, label in zip(x, labels):
        ax.text(
            xi, 0.5, label,
            ha="center", va="center", fontsize=12,
            bbox=dict(boxstyle="round", fc="white")
        )

    for i in range(len(labels) - 1):
        ax.annotate(
            "",
            xy=(x[i + 1] - 0.28, 0.5),
            xytext=(x[i] + 0.28, 0.5),
            arrowprops=dict(arrowstyle="->", lw=2),
        )

    ax.axis("off")
    ax.set_xlim(-0.5, len(labels) - 0.5)
    ax.set_ylim(0, 1)

    plt.tight_layout()
    filename = os.path.join(outdir, "FigureS2_hydraulic_analogy.png")
    plt.savefig(filename, dpi=300)
    plt.close()

    return filename


# ============================================================
# Main
# ============================================================

def main():
    outdir = "figures"
    os.makedirs(outdir, exist_ok=True)

    params = {
        "t0": 0.0,
        "tf": 160.0,
        "dt": 0.05,

        "C0": 8.0,
        "M0": 5.0,
        "v0": 0.005,

        "A0": 0.45,
        "A_amp": 0.35,
        "period": 24.0,

        "K_C": 6.0,

        "v_base": 0.004,
        "v_max": 0.055,
        "tau_v": 18.0,
        "t_dev": 45.0,
        "k_dev": 0.12,

        "Yg": 0.75,
        "k_loss": 0.025,
    }

    res = simulate(params)

    fig2 = plot_dynamic_system(res, outdir=outdir)
    fig3 = plot_counterexample(outdir=outdir)
    fig4 = plot_scenarios(outdir=outdir)
    figS1 = plot_mechanical_analogy(outdir=outdir)
    figS2 = plot_hydraulic_analogy(outdir=outdir)

    idx_peak = np.argmax(res["dPi_dt"])

    print("=== Sink momentum ODE mini-model ===")
    print(f"Final C:       {res['C'][-1]:.3f} gC")
    print(f"Final M:       {res['M'][-1]:.3f} gC")
    print(f"Final v:       {res['v'][-1]:.5f} mol s-1")
    print(f"Final Pi:      {res['Pi'][-1]:.3f} gC mol s-1")
    print()
    print("Peak dynamic sink strength:")
    print(f"Time:          {res['t'][idx_peak]:.2f} s")
    print(f"dPi/dt:        {res['dPi_dt'][idx_peak]:.5f} gC mol s-2")
    print(f"Structural:    {res['structural_term'][idx_peak]:.5f}")
    print(f"Flow accel.:   {res['activation_term'][idx_peak]:.5f}")
    print()
    print("Saved figures:")
    print(fig2)
    print(fig3)
    print(fig4)
    print(figS1)
    print(figS2)
    print(f"Separate Figure 2 panels saved in: {outdir}/")


if __name__ == "__main__":
    main()