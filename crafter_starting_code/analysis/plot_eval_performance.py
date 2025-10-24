import argparse
import pathlib
import pickle

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def read_pkl(path):
    events = []
    with (open(path, "rb")) as openfile:
        while True:
            try:
                events.append(pickle.load(openfile))
            except EOFError:
                break
    return events


"""def read_crafter_logs(indir, clip=True):
    indir = pathlib.Path(indir)
    # read the pickles
    filenames = sorted(list(indir.glob("**/*/eval_stats.pkl")))
    runs = []
    for idx, fn in enumerate(filenames):
        df = pd.DataFrame(columns=["step", "avg_return"], data=read_pkl(fn))
        df["run"] = idx
        runs.append(df)

    # some runs might not have finished and you might want to clip all of them
    # to the shortest one.
    if clip:
        min_len = min([len(run) for run in runs])
        runs = [run[:min_len] for run in runs]
        print(f"Clipped al runs to {min_len}.")

    # plot
    df = pd.concat(runs, ignore_index=True)
    sns.lineplot(x="step", y="avg_return", data=df)
    plt.savefig("demo_plot.png")
    plt.show()  """


def read_crafter_logs(indirs):  # [FR] Le paramètre est maintenant 'indirs' (pluriel). / [EN] The parameter is now 'indirs' (plural).
    """
    [FR] Lit les logs de plusieurs types d'agents, les agrège et trace le graphique comparatif.
    [EN] Reads logs from multiple agent types, aggregates them, and plots the comparison graph.
    """
    runs = []

    # --- DÉBUT DE LA CORRECTION ---
    # [FR] On boucle sur chaque chemin de dossier reçu dans la liste 'indirs'.
    # [EN] We loop over each directory path received in the 'indirs' list.
    for indir in indirs:
        indir = pathlib.Path(
            indir)  # [FR] Convertit chaque chemin individuellement. / [EN] Converts each path individually.

        filenames = sorted(list(indir.glob("**/*/eval_stats.pkl")))

        if not filenames:
            print(f"Warning: No 'eval_stats.pkl' files found in {indir}")
            continue

        for idx, fn in enumerate(filenames):
            df = pd.DataFrame(columns=["step", "avg_return"], data=read_pkl(fn))
            df["run"] = idx
            df["agent_name"] = indir.name
            runs.append(df)
    # --- FIN DE LA CORRECTION ---

    if not runs:
        print("Error: No data loaded. Exiting.")
        return

    df = pd.concat(runs, ignore_index=True)

    sns.lineplot(x="step", y="avg_return", hue="agent_name", data=df, errorbar='sd')

    plt.title("Agent Performance Comparison")
    plt.xlabel("Training Steps")
    plt.ylabel("Average Episodic Return")

    plt.savefig("performance_plot.png")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    """
    parser.add_argument(
        "--logdir",
        default="logdir/random_agent",
        help="Path to the folder containing different runs.",
    )"""
    parser.add_argument(
        "--logdir",
        nargs='+',  # <-- C'est la ligne clé qui permet de prendre plusieurs arguments
        required=True,
        help="Path(s) to folder(s) containing different runs.",
    )
    cfg = parser.parse_args()

    read_crafter_logs(cfg.logdir)
