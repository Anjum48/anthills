import numpy as np
import pandas as pd
import pymc3 as pm
import scipy as sp
import seaborn as sns
import lasio
import os
import statsmodels.tsa.stattools as ts
from time import time
from theano import tensor as T
from matplotlib import pyplot as plt
from scipy import stats
from scipy.misc import derivative
from scipy.interpolate import splrep, splev
from multiprocessing import Pool
__author__ = 'sayea5'
#  Adapted from http://austinrochford.com/posts/2016-02-25-density-estimation-dpm.html


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
        super().__init__(a=np.log10(t2_min), b=np.log10(t2_max))
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
    def __init__(self, las_file, t2_min=0.3, t2_max=6000, plots=False):
        self.las_file = las_file
        self.t2_min = t2_min
        self.t2_max = t2_max
        self.t2_cutoff = 1
        self.plots = plots
        self.df = lasio.read("R-540_NMR.las").df().dropna()
        self.bins = len(self.df.columns)
        self.k = 20
        self.N = 10000
        self.max_iterations = 10000
        self.mini_batch = 1000
        self.trace = None  # Using the previous trace as a starting point for the next won't work in parallel runs

        # Create directories to save plots
        if not os.path.exists(os.path.join(os.pardir, "NMRGP_Output")):
            os.makedirs(os.path.join(os.pardir, "NMRGP_Output"))
        if not os.path.exists(os.path.join(os.pardir, "NMRGP_Output", "plots")):
            os.makedirs(os.path.join(os.pardir, "NMRGP_Output", "plots"))

    def density_estimation(self, data, iters):
        with pm.Model() as model:
            alpha = pm.Gamma('alpha', 1., 1.)
            beta = pm.Beta('beta', 1., alpha, shape=self.k)
            w = pm.Deterministic('w', beta * T.concatenate([[1], T.extra_ops.cumprod(1 - beta)[:-1]]))
            component = pm.Categorical('component', w, shape=len(data))

            std_dev = pm.HalfNormal(name="std_dev", sd=1, shape=self.k)
            mu = pm.Normal('mu', 0, std_dev, shape=self.k)
            obs = np.log(pm.Normal('obs', mu=mu[component], sd=std_dev[component], observed=data))

        with model:
            step1 = pm.Metropolis(vars=[alpha, beta, w, std_dev, mu, obs])
            # step2 = pm.CategoricalGibbsMetropolis(vars=[component])
            step2 = pm.ElemwiseCategorical([component], np.arange(self.k))  # This is depreciated but faster
            trace_ = pm.sample(iters, [step1, step2], progressbar=True, trace=self.trace)
        return trace_

    def process_depth(self, df_row):
        print("Processing depth:", df_row[0])
        t2_depth = df_row[1]
        t2 = T2Dist(t2_min=self.t2_min, t2_max=self.t2_max, bins=self.bins, t2_data=t2_depth, clip_point=self.t2_cutoff)
        data = t2.rvs(size=self.N)
        t2_mean = np.mean(data)
        t2_std = np.std(data)
        data_std = (data - t2_mean) / t2_std

        # Standard method without early stopping
        self.trace = self.density_estimation(data=data_std, iters=self.max_iterations)
        self.trace = self.trace[int(self.max_iterations/2)::10]

        # # Run the MCMC model and stop early in case of convergence by checking if the traces from all of the means are
        # # stationary using the Ljung-Box test. This doesn't work so well
        # i = 0
        # while i < self.max_iterations:
        #     self.trace = self.density_estimation(data=data_std, iters=self.mini_batch)
        #     thinned_trace = self.trace[int((i + self.mini_batch) / 2)::5]
        #     test_results = []
        #     for j in range(self.k):
        #         acf_results = ts.acf(thinned_trace['mu'][:, j], nlags=10, qstat=True)
        #         if j == 0:
        #             # print(thinned_trace['mu'][:, j])
        #             # print(acf_results[1])
        #             # print(acf_results[2])
        #             print(acf_results[1] < acf_results[2])
        #         test_results.append(all(acf_results[1] < acf_results[2]))
        #     i += self.mini_batch
        #     print(i, sum(test_results))
        #     if sum(test_results) == self.k:
        #         print("Stopped early after", i, "iterations")
        #         break
        #
        # # Thin the final traces
        # self.trace = self.trace[int(i/2)::5]

        if self.plots:
            self.make_plots(data, t2_dist=t2, t2_mean=t2_mean, t2_std=t2_std, depth=t2_depth.name)

        # Find stats
        weights = (self.trace['w'][:, np.newaxis, :].mean(axis=0))
        means = (self.trace['mu'][:, np.newaxis, :].mean(axis=0)) * t2_std + t2_mean
        std_devs = (self.trace['std_dev'][:, np.newaxis, :].mean(axis=0)) * t2_std
        weights_error = (self.trace['w'][:, np.newaxis, :].std(axis=0))
        means_error = (self.trace['mu'][:, np.newaxis, :].std(axis=0))
        std_devs_error = (self.trace['std_dev'][:, np.newaxis, :].std(axis=0))

        # Exclude distributions with < 1% contribution and renormalise
        weights[weights < 0.01] = 0
        weights = weights / np.sum(weights)
        means[weights == 0] = 0
        std_devs[weights == 0] = 0

        # Sort by increasing mean T2
        sort_order = np.argsort(means[0])
        final_result = (t2_depth.name,)
        for statistic in [weights, means, std_devs, weights_error, means_error, std_devs_error]:
            final_result += (statistic[0][sort_order],)

        return final_result

    def make_plots(self, data, t2_dist, t2_mean, t2_std, depth):
        x_plot = np.linspace(np.log10(self.t2_min), np.log10(self.t2_max), 200)
        t2_pdf = t2_dist.pdf(x_plot)

        # Make trace plots
        pm.traceplot(self.trace)
        plt.savefig(os.path.join(os.pardir, "NMRGP_Output", "plots", "Traces_" + str(depth) + ".png"))

        # Make autocorrelation plots for the traces
        pm.autocorrplot(self.trace, varnames=['mu'])
        plt.savefig(os.path.join(os.pardir, "NMRGP_Output", "plots", "Autocorr_" + str(depth) + ".png"))

        n_components_used = np.apply_along_axis(lambda x: np.unique(x).size, 1, self.trace['component'])
        sns.set(color_codes=True)

        # We plot the distribution of the number of mixture components used
        fig1, ax1 = plt.subplots(figsize=(8, 6))
        bins = np.arange(n_components_used.min(), n_components_used.max() + 1)
        ax1.hist(n_components_used + 1, bins=bins, normed=True, lw=0, alpha=0.75)
        ax1.set_xticks(bins + 0.5)
        ax1.set_xticklabels(bins)
        ax1.set_xlim(bins.min(), bins.max() + 1)
        ax1.set_xlabel('Number of mixture components used')
        ax1.set_ylabel('Posterior probability')
        ax1.set_title(str(depth) + "mMD")
        fig1.savefig(os.path.join(os.pardir, "NMRGP_Output", "plots", "Components_" + str(depth) + ".png"))

        # We now compute and plot our posterior density estimate.
        post_pdf_contribs = sp.stats.norm.pdf(np.atleast_3d(x_plot),
                                              self.trace['mu'][:, np.newaxis, :] * t2_std + t2_mean,
                                              self.trace['std_dev'][:, np.newaxis, :] * t2_std)
        post_pdfs = (self.trace['w'][:, np.newaxis, :] * post_pdf_contribs).sum(axis=-1)
        post_pdf_low, post_pdf_high = np.percentile(post_pdfs, [2.5, 97.5], axis=0)

        fig2, ax2 = plt.subplots(figsize=(8, 6))
        ax2.hist(data, bins=self.bins, normed=True, lw=0, alpha=0.5)
        ax2.fill_between(x_plot, post_pdf_low, post_pdf_high, color='gray', alpha=0.45)
        ax2.plot(x_plot, post_pdfs[0], color='gray', label='Posterior sample densities')
        ax2.plot(x_plot, post_pdfs[::100].T, color='gray')
        ax2.plot(x_plot, post_pdfs.mean(axis=0), color='k', label='Posterior expected density')
        ax2.plot(x_plot, t2_pdf, color='r', label="NMR T2 Data")
        ax2.set_xticklabels(10 ** np.array(ax2.get_xticks().tolist()))
        ax2.set_xlabel('$T_{2} (ms)$')
        ax2.set_yticklabels([])
        ax2.set_ylabel('Density')
        ax2.set_title(str(depth) + "mMD")
        ax2.legend(loc=2)
        fig2.savefig(os.path.join(os.pardir, "NMRGP_Output", "plots", "Posterior_Densities_" + str(depth) + ".png"))

        # We can decompose this density estimate into its (weighted) mixture components.
        fig3, ax3 = plt.subplots(figsize=(8, 6))
        ax3.hist(data, bins=self.bins, normed=True, lw=0, alpha=0.5)
        ax3.plot(x_plot, post_pdfs.mean(axis=0), color='k', label='Posterior expected density')
        ax3.plot(x_plot, (self.trace['w'][:, np.newaxis, :] * post_pdf_contribs).mean(axis=0)[:, 0],
                 '--', color='k', label='Posterior expected mixture\ncomponents\n(weighted)')
        ax3.plot(x_plot, (self.trace['w'][:, np.newaxis, :] * post_pdf_contribs).mean(axis=0), '--', color='k')
        ax3.plot(x_plot, t2_pdf, color='r', label="NMR T2 Data")
        ax3.set_xticklabels(10 ** np.array(ax3.get_xticks().tolist()))
        ax3.set_xlabel('$T_{2} (ms)$')
        ax3.set_yticklabels([])
        ax3.set_ylabel('Density')
        ax3.set_title(str(depth) + "mMD")
        ax3.legend(loc=2)
        fig3.savefig(os.path.join(os.pardir, "NMRGP_Output", "plots", "Posterior_Components_" + str(depth) + ".png"))
        # plt.show()
        plt.close()

    def run(self, start_depth=None, end_depth=None, cpus=1):
        self.df = self.df.loc[start_depth:end_depth]
        p = Pool(cpus)  # Number of CPU processes
        output = p.map(self.process_depth, self.df.iterrows())
        return output


if __name__ == "__main__":
    filename = "R-540_NMR.las"
    start = time()
    np.random.seed(12345)  # set random seed for reproducibility
    gp = NMRGaussianProcess(filename)
    output = gp.run(start_depth=2350, end_depth=2400, cpus=1)
    print("Finished batch run in", (time() - start) / 3600, "hours")

    # Process the output and save to CSV
    depths = []
    weights = []
    means = []
    std_devs = []
    weights_error = []
    means_error = []
    std_devs_error = []

    for result in output:
        depths.append(result[0])
        weights.append(result[1])
        means.append(result[2])
        std_devs.append(result[3])
        weights_error.append(result[4])
        means_error.append(result[5])
        std_devs_error.append(result[6])

    weights = pd.DataFrame(data=weights, index=depths)
    means = pd.DataFrame(data=means, index=depths)
    std_devs = pd.DataFrame(data=std_devs, index=depths)
    depths = weights.index.values

    weights.to_csv(filename[:-4]+"_weights.csv")
    means.to_csv(filename[:-4]+"_means.csv")
    std_devs.to_csv(filename[:-4]+"_std_devs.csv")
    pd.DataFrame(data=weights_error, index=depths).to_csv(filename[:-4]+"_weights_error.csv")
    pd.DataFrame(data=means_error, index=depths).to_csv(filename[:-4]+"_means_error.csv")
    pd.DataFrame(data=std_devs_error, index=depths).to_csv(filename[:-4]+"_std_devs_error.csv")
    print("Saved outputs to CSV")

    # TODO: Get rid of this - Make a separate Techlog formatter. Also calculate permeability
    step = (np.log10(gp.t2_max) - np.log10(gp.t2_min))/(gp.bins - 1)
    log_time = [np.log10(gp.t2_min) + x*step for x in range(gp.bins)]
    max_t2 = gp.df.max(axis=1)

    compilation = []
    for depth in depths:
        x = []
        for i in range(gp.k):
            if np.sum(weights.loc[depth][i]) > 0:
                dist = stats.norm.pdf(log_time, means.loc[depth][i], std_devs.loc[depth][i])*weights.loc[depth][i]
                dist = dist*max_t2.loc[depth]  # rescale T2 distribution to the measured T2
                x.append(dist)
            else:
                x.append(np.zeros(gp.bins))
        compilation.append(np.array(x))

    # Techlog Formatting
    long_data = []
    for i, depth in enumerate(depths):
        for j in range(gp.bins):
            long_data.append([depth] + list(compilation[i][:, j]))

    df = pd.DataFrame(long_data, columns=["MD"] + ["T2_Component_"+str(x) for x in range(gp.k)])
    df.to_csv(filename[:-4]+"_T2_Components.csv")
