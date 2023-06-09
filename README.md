# NMR Pore typing using Bayesian Gaussian mixture models
A tool for fitting Gaussian distributions to NMR T2 distributions to allow pore typing in carbonates using Bayesian Gaussian mixture models.
Since scikit-learn 0.18, `BayesianGaussianMixture` is much faster than the PyMC3 implementation

# Instructions to use:
1. Clone the repository
2. In the same folder location as the repository, create a folder called `input_files` where the LAS files will go
3. Each LAS file should have the TCMR curve and a T2 distribution. T2_MIN & T2_MAX should be 0.3 & 6000ms
4. Run `anthills_sklearn.py`
5. The outputs will be populated in a folder called `anthills_Output`
6. The calculated permeability curve can be loaded to Techlog via a CSV file