"""Matplotlib rendering of simulation states (screenshots) and trajectories."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D

# ROAD, BUILDING, HAZARD, GRATE, SAFE, STATION
COLOURS = ["#e8e8e8", "#555b66", "#f08a24", "#b9a7d6", "#6fcf97", "#6fb7e8"]
CMAP = ListedColormap(COLOURS)
AGENT_STYLE = {"JOHN": ("o", "#1f5fbf", "John"),
               "T-800": ("s", "#1b7a1b", "T-800"),
               "T-1000": ("^", "#c4161c", "T-1000")}


def draw(ax, sim, title="", trails=True, belief=None, show_fog=False, legend=True):
    w = sim.world
    ax.imshow(w.cells, cmap=CMAP, vmin=0, vmax=5, origin="upper", interpolation="nearest")
    if belief is not None:
        import numpy as np  # matplotlib dependency, always present
        ax.imshow(np.array(belief), cmap="Reds", alpha=0.45, origin="upper",
                  interpolation="nearest")
    if show_fog:
        j = sim.john
        for c in w.visible_cells(j.pos, j.radius):
            ax.add_patch(plt.Rectangle((c[0] - .5, c[1] - .5), 1, 1, fill=False,
                                       ec="#1f5fbf", lw=0.3, alpha=0.5))
    if trails:
        for name, pts in sim.traj.items():
            m, col, _ = AGENT_STYLE[name]
            ax.plot([p[0] for p in pts], [p[1] for p in pts], "-", color=col, lw=1.3, alpha=0.65)
    for a in sim.agents:
        m, col, lab = AGENT_STYLE[a.name]
        ax.plot(a.pos[0], a.pos[1], m, color=col, ms=9, mec="white", mew=1.2,
                alpha=1.0 if a.alive else 0.3)
    ax.set_xticks(range(0, w.size, 5))
    ax.set_yticks(range(0, w.size, 5))
    ax.set_xlim(-.5, w.size - .5)
    ax.set_ylim(w.size - .5, -.5)
    ax.grid(True, color="white", lw=0.3, alpha=0.5)
    handles = [Line2D([], [], marker=AGENT_STYLE[n][0], color="w", markerfacecolor=AGENT_STYLE[n][1],
                     ms=8, label=AGENT_STYLE[n][2]) for n in AGENT_STYLE]
    handles += [Line2D([], [], marker="s", color="w", markerfacecolor=COLOURS[i], ms=8, label=l)
               for i, l in [(1, "Building"), (2, "Hazard"), (3, "Fence (T-1000 only)"),
                            (4, "Safe zone"), (5, "Station")]]
    if legend:
        ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=7,
              frameon=False)
    ax.set_title(title or f"{sim.cfg.name} seed {sim.seed} step {sim.step_no}", fontsize=9)


def render(sim, path, title="", trails=True, belief=None, show_fog=False):
    fig, ax = plt.subplots(figsize=(6.4, 6.4), dpi=110)
    draw(ax, sim, title, trails, belief, show_fog)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
