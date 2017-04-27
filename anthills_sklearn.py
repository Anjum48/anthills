import numpy as np
import pandas as pd
import seaborn as sns
import lasio
import os
from time import time
from collections import OrderedDict
from matplotlib import pyplot as plt
from scipy import stats
from scipy.misc import derivative
from scipy.interpolate import splrep, splev
from sklearn.mixture import BayesianGaussianMixture
from multiprocessing import Pool
__author__ = 'sayea5'
#  Adapted from http://scikit-learn.org/stable/modules/mixture.html#variational-bayesian-gaussian-mixture


class T2Dist(stats.rv_continuous):
    def __init__(self, t2_min, t2_max, bins, t2_data, clip_point=None):
        """
        Creates a continuous T2 distribution which can be sampled from
        Binned T2 data is converted to a continuous variable using a cubic spline
        :param t2_min: Min extent of T2 e.g. 0.3ms
        :param t2_max: Max extent of T2 e.g. 30000ms
        :param bins: Number of bins in the T2 distribution from the LAS file
        :param t2_data: The T2 values from the LAS file (array of len bins)
        """
        stats.rv_continuous.__init__(self, a=np.log10(t2_min), b=np.log10(t2_max))
        time_step = (np.log10(t2_max) - np.log10(t2_min)) / (bins - 1)
        self.log_time = [np.log10(t2_min) + x * time_step for x in range(bins)]
        if clip_point is None:
            self.t2 = t2_data
        else:
            self.t2 = self.clip_data(clip_point, t2_data)

    def _cdf(self, x, *t2_data):
        tck = splrep(self.log_time, self.t2.cumsum()/self.t2.cumsum().max())
        return splev(x, tck)

    def _pdf(self, x, *args):
        d = derivative(self._cdf, x, dx=1e-5, args=args, order=5)
        d[d < 0] = 0
        return d

    def clip_data(self, clip_point, t2_data):
        clipped_t2_data = []
        for x, y in zip(self.log_time, t2_data):
            if x < np.log10(clip_point):
                clipped_t2_data.append(1e-8)
            else:
                clipped_t2_data.append(y)
        return np.array(clipped_t2_data)


class NMRGaussianProcess:
    def __init__(self, las_file, well, output_path, t2_min=0.3, t2_max=6000, plots=False, seed=None):
        self.t2_min = t2_min
        self.t2_max = t2_max
        self.t2_cutoff = 0.3  # T2 values to exclude below
        self.plots = plots
        self.df = lasio.read(las_file).df().dropna()
        # TODO: Feed porosity curve name (TCMR) as an input
        self.phit = self.df["TCMR"].values.reshape((len(self.df), 1))
        del self.df["TCMR"]
        self.well = well
        self.bins = len(self.df.columns)
        self.k = 20
        self.N = 10000
        self.max_iterations = 20000
        self.depth_count = 0
        self.seed = seed
        self.output_path = output_path
        self.gm = BayesianGaussianMixture(n_components=self.k, degrees_of_freedom_prior=self.k,
                                          max_iter=self.max_iterations, warm_start=False, random_state=self.seed)

        # Create directories to save outputs
        if not os.path.exists(os.path.join(self.output_path, "anthills_output")):
            os.makedirs(os.path.join(self.output_path, "anthills_output"))
        if not os.path.exists(os.path.join(self.output_path, "anthills_output", self.well)) and self.plots:
            os.makedirs(os.path.join(self.output_path, "anthills_output", self.well))
        if not os.path.exists(os.path.join(self.output_path, "anthills_output", "CSV")):
            os.makedirs(os.path.join(self.output_path, "anthills_output", "CSV"))
        if not os.path.exists(os.path.join(output_folder, "anthills_output", "Permeability")):
            os.makedirs(os.path.join(output_folder, "anthills_output", "Permeability"))

    def process_depth(self, df_row):
        depth_start = time()
        t2_depth = df_row[1]
        t2 = T2Dist(t2_min=self.t2_min, t2_max=self.t2_max, bins=self.bins, t2_data=t2_depth, clip_point=self.t2_cutoff)
        data = t2.rvs(size=self.N, random_state=self.seed)
        self.gm.fit(data.reshape(-1, 1))  # Note that this uses half of all available CPUs
        self.depth_count += 1
        print(self.well, "Processed depth:", df_row[0], "in", int(time() - depth_start), "seconds",
              "("+str(self.depth_count)+"/"+str(len(self.df))+")")

        if self.plots:
            self.make_plots(t2, data, df_row[0])

    def make_plots(self, t2_dist, data, depth):
        x_plot = np.linspace(np.log10(self.t2_min), np.log10(self.t2_max), 200)
        t2_pdf = t2_dist.pdf(x_plot)

        components = []
        for i in range(self.k):
            components.append(self.gm.weights_[i] * stats.norm.pdf(x_plot, self.gm.means_[i],
                                                                   np.sqrt(self.gm.covariances_[i])).flatten())
        components = np.transpose(components)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.hist(data, bins=64, normed=True, lw=0, alpha=0.5)
        ax.plot(x_plot, np.sum(components, axis=1), color='k', label='Posterior expected density')
        ax.plot(x_plot, components, '--', color='k', label='Posterior expected mixture\ncomponents\n(weighted)')
        ax.plot(x_plot, t2_pdf, color='r', label="NMR T2 Data")
        ax.set_xticklabels(10 ** np.array(ax.get_xticks().tolist()))
        ax.set_xlabel('$T_{2} (ms)$')
        ax.set_yticklabels([])
        ax.set_ylabel('Density')
        ax.set_title(self.well + " " + str(depth) + "mMD")
        ax.legend(loc=2)
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = OrderedDict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())
        fig.savefig(os.path.join(self.output_path, "anthills_output", self.well, self.well + "_" + str(depth) + ".png"))
        # plt.show()
        plt.close()

    def run(self, start_depth=None, end_depth=None):
        self.df = self.df.loc[start_depth:end_depth]
        means = []
        variances = []
        weights = []
        for row in self.df.iterrows():
            self.process_depth(row)
            means.append(self.gm.means_.flatten())
            variances.append(self.gm.covariances_.flatten())
            weights.append(self.gm.weights_.flatten())

        means = pd.DataFrame(data=np.array(means), index=self.df.index)
        variances = pd.DataFrame(data=np.array(variances), index=self.df.index)
        weights = pd.DataFrame(data=np.array(weights), index=self.df.index)

        # Calculate permeability using modified SDR component model
        perm = (((self.phit * weights) ** 4) * ((10 ** means) ** 2)).sum(axis=1)
        perm.rename("GMM_Perm", inplace=True)

        # Save outputs
        means.to_csv(os.path.join(self.output_path, "anthills_output", "CSV", self.well+"_means.csv"))
        variances.to_csv(os.path.join(self.output_path, "anthills_output", "CSV", self.well+"_variances.csv"))
        weights.to_csv(os.path.join(self.output_path, "anthills_output", "CSV", self.well+"_weights.csv"))
        perm.to_csv(os.path.join(output_folder, "anthills_output", "Permeability", self.well+"_anthills.csv"), header=True)


def multiprocessing_helper(las_file):
    start = time()
    path = os.path.join(os.pardir, "Input_files", las_file)
    well = las_file[:5]
    np.random.seed(12345)
    NMRGaussianProcess(path, well, output_path=output_folder, plots=True, seed=12345).run()
    print(well, "Finished run in", round((time() - start) / 3600, 2), "hours")

output_folder = os.pardir
# output_folder = "/hpcdata/sayea5"

if __name__ == "__main__":
    # Create folder for saving outputs
    if not os.path.exists(os.path.join(output_folder, "anthills_output")):
        os.makedirs(os.path.join(output_folder, "anthills_output"))

    input_folder = os.path.join(os.pardir, "Input_files")
    las_files = [f for f in os.listdir(input_folder) if os.path.isfile(os.path.join(input_folder, f))]

    # for las_file in las_files:
    #     file_name = os.path.join(os.pardir, "Input_files", las_file)
    #     well_name = las_file[:5]
    #     np.random.seed(12345)
    #     NMRGaussianProcess(file_name, well_name, plots=True, seed=12345).run()

    # p = Pool()
    p = Pool(1)  # gm.fit() spawns processes = number to CPU cores, i.e. 12 threads on a 24 thread machine
    output = p.map(multiprocessing_helper, las_files)
