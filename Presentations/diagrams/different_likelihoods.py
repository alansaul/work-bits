import scipy as sp
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
sns.set_context('talk')
sns.set_style('white')


# Produce a grid of different likelihood distributions
mean = 2.0
scale = 3.0

# Gaussian, Log Gaussian, Student T, Bernoulli, Beta, Poisson,
distributions = [(stats.norm(loc=mean, scale=scale), True, 'Gaussian $N(y|\mu=f, \sigma=3)$'),
                 (stats.lognorm(s=0.7), True, 'Log-Gaussian $LG(y|\mu=f, \sigma=0.7)$'),
                 (stats.t(loc=mean, scale=scale, df=4.0), True, 'Student-T $t(y|\mu=f, \sigma=3, df=4)$'),
                 (stats.beta(a=mean, b=mean*0.8), True, 'Beta $Be(y|a=f, b=1.6)$'),
                 (stats.bernoulli(p=0.3), False, 'Bernoulli $B(y|p=f)$'),
                 (stats.poisson(mu=mean), False, 'Poisson $P(y|\lambda=f)$')
                 ]


fig, axes = plt.subplots(2, 3, figsize=(15, 15))

for ax, (dist, continuous, dist_name) in zip(axes.flatten(), distributions):
    if continuous:
        x = np.linspace(dist.ppf(0.0001),
                        dist.ppf(0.9999),
                        300
                        )
        ax.plot(x, dist.pdf(x), 'b-')
    else:
        x = np.arange(dist.ppf(0.00001),
                    dist.ppf(0.99999)
                    )
        if x.shape[0] == 1:  # bernoulli
            x = np.array([0.0, 1.0])
        ax.plot(x, dist.pmf(x), 'bo', ms=8)
        ax.vlines(x, 0, dist.pmf(x), colors='b', lw=5, alpha=0.5)

    ax.grid(False)
    ax.title.set_text(dist_name)
    sns.despine()
    ax.set_ylim(bottom=0.0)
    ax.set_xlabel('y')
    ax.set_ylabel('p(y|f)')

plt.tight_layout(pad=2.5, h_pad=5.0)
plt.show()
fig.savefig('all_likelihoods.pdf')


# lik_f = np.nan_to_num(np.array([l.pdf(np.atleast_2d(f), np.ones((1,1))*y, meta) for f in ff])[:,:,0])
# lik_f = nan_to_ninf(lik_f)
# log_lik_f = nan_to_ninf(np.log(lik_f))
# #Plot the likelihood
# ax1.plot(ff, log_lik_f, label='log likelihood, $\log p(y=4|f)$', lw=lw)
# ax1.legend()
# ax1.set_ylim(log_prior_f.min(), log_prior_f.max()+3)
