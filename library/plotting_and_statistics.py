import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
from scipy import stats
import numpy as np
import pandas as pd
import seaborn as sns


def perform_task_on_bar_dict(bar_dict, function):
    function_dict = {}
    for bar_name in bar_dict:
        function_dict[bar_name] = function(bar_dict[bar_name])
    return function_dict


def plot_autocorrelation(bars):
    bar_returns = np.log(bars["close"]).diff().dropna()
    plot_acf(bar_returns, lags=10, zero=False)
    plt.title("Bar AutoCorrelation")
    plt.show()


def plot_bars_histogram_close(bars):
    plt.figure(figsize=(15, 8))
    plt.hist(
        bars.close.pct_change().dropna().values.tolist()[: len(bars)],
        label="Price bars",
        alpha=0.5,
        normed=True,
        bins=50,
        range=(-0.01, 0.01),
    )
    plt.legend()
    plt.show()


def get_bar_stats(bars):
    autocorr = pd.Series.autocorr(bars.close.pct_change().dropna()[: len(bars)])
    variance = np.var(bars.close.pct_change().dropna()[: len(bars)])
    jarque_bera_stat = stats.jarque_bera(bars.close.pct_change().dropna()[: len(bars)])
    shapiro_stat = stats.shapiro(bars.close.pct_change().dropna()[: len(bars)])
    return autocorr, variance, jarque_bera_stat, shapiro_stat


def plot_bars_timeseries_close(bars):
    plt.figure()
    plt.plot(bars.close[: int(len(bars) * 0.5)])
    plt.plot(bars.close[int(len(bars) * 0.5) : int(len(bars) * 0.7)])
    plt.show()


def get_bars_from_list(bars):
    bar_dict = {}
    for bar in bars:
        bar_dict[bar] = pd.read_csv(
            "data/" + bar + "_bars.csv", index_col=0, parse_dates=True
        )
    return bar_dict


# split this into a get_bars function aswell
def measure_bar_stability(bars, sample_period):
    bar_list = {}
    count_list = {}
    for bar in bars:
        bar_list[bar] = pd.read_csv(
            "data/" + bar + "_bars.csv", index_col=0, parse_dates=True
        )
        count_list[bar] = (
            bar_list[bar]["close"].resample(sample_period, label="right").count()
        )

    count_df = pd.concat(count_list, axis=1)
    count_df.columns = bars

    count_df.loc[:, bars].plot(
        kind="bar", figsize=[25, 5], color=("darkred", "darkblue", "green", "darkcyan")
    )
    plt.title(
        "Number of bars over time",
        loc="center",
        fontsize=20,
        fontweight="bold",
        fontname="Times New Roman",
    )
    plt.show()


def check_normality_of_bars(bar_dict):
    print("Test Statistics:")
    plt.figure(figsize=(16, 12))
    for bar_name in bar_dict:
        return_dict = {}
        standard_dict = {}
        return_dict[bar_name] = np.log(bar_dict[bar_name]["close"]).diff().dropna()
        print(bar_name, "\t", int(stats.jarque_bera(return_dict[bar_name])[0]))
        standard_dict[bar_name] = (
            return_dict[bar_name] - return_dict[bar_name].mean()
        ) / return_dict[bar_name].std()
        sns.kdeplot(standard_dict[bar_name], label=bar_name, color=np.random.rand(3))
    sns.kdeplot(
        np.random.normal(size=1000000), label="Normal", color="white", linestyle="--"
    )
    plt.xticks(range(-5, 6))
    plt.legend(loc=8, ncol=5)
    plt.title(
        "Exhibit 1 - Partial recovery of Normality through a price sampling process \nsubordinated to a volume, tick, dollar clock",
        loc="center",
        fontsize=20,
        fontweight="bold",
        fontname="Times New Roman",
    )
    plt.xlim(-5, 5)
    plt.show()


def plot_learning_curves(loss, val_loss):
    plt.plot(np.arange(len(loss)) + 0.5, loss, "b.-", label="Training loss")
    plt.plot(np.arange(len(val_loss)) + 1, val_loss, "r.-", label="Validation loss")
    plt.gca().xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))
    plt.axis([1, 20, 0, 2])
    plt.legend(fontsize=14)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.grid(True)

