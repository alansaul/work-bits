import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import GPy
import seaborn as sns
plt.close('all')

sns.set()
sns.set_context('talk')
sns.set_style('white')
#colors = seaborn.color_palette('Set2', 2)

np.random.seed(1)
#Poisson example
X = np.linspace(0,23.50,48)[:, None]
fs = 3*np.sin(10 + 0.6*X) + np.sin(0.1*X)
intensities = fs + 4
Yp = np.array([sp.random.poisson(intensity) for intensity in intensities])
Yl = np.array([sp.random.lognormal(np.sqrt(intensity)*0.1, sigma=2.0) for intensity in intensities])
Yb = np.array([sp.random.binomial(p=sp.stats.norm.cdf(f), n=1) for f in fs])

def plot_samples(times, samples, ax, dist_name):
    ax.plot(times, samples, 'bo')
    ax.title.set_text(dist_name)

fig, axes = plt.subplots(1,3, figsize=(15, 5))

plot_samples(X, Yp, axes[0], dist_name='Poisson')
plot_samples(X, Yl, axes[1], dist_name='Log-Gaussian')
plot_samples(X, Yb, axes[2], dist_name='Bernoulli')
plt.tight_layout(h_pad=3.0)
fig.savefig("datatypes.pdf")
plt.show()

