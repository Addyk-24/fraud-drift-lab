import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src import config

COLORS = {"blue": "#2b6cb0", "red": "#c53030", "green": "#2f855a", "orange": "#dd6b20",
          "gray": "#718096", "purple": "#6b46c1"}


def setup():
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 200, "savefig.bbox": "tight",
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
        "grid.alpha": 0.25, "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
        "legend.frameon": False,
    })


def save(fig, name):
    config.FIG_DIR.mkdir(parents=True, exist_ok=True)
    p = config.FIG_DIR / f"{name}.png"
    fig.savefig(p)
    plt.close(fig)
    print("figure ->", p)
    return p


setup()
