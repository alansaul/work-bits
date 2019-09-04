import scipy as sp
from scipy import stats
from scipy import integrate
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
sns.set_context('talk')
sns.set_style('white')
plt.close('all')


# Produce a grid of different likelihood distributions
mean = 2.0
scale = 3.0

# Gaussian, Log Gaussian, Student T, Bernoulli, Beta, Poisson,
distributions = [(stats.norm(loc=mean, scale=scale), True, 'Gaussian $N(y|\mu=f=2, \sigma=3)$'),
                 (stats.lognorm(s=0.7), True, 'Log-Gaussian $LG(y|\mu=f=2, \sigma=0.7)$'),
                 (stats.t(loc=mean, scale=scale, df=4.0), True, 'Student-T $t(y|\mu=f=2, \sigma=3, df=4)$'),
                 (stats.beta(a=mean, b=mean*0.8), True, 'Beta $Be(y|a=f=2, b=1.6)$'),
                 (stats.bernoulli(p=0.3), False, 'Bernoulli $B(y|p=f=0.3)$'),
                 (stats.poisson(mu=mean), False, 'Poisson $P(y|\lambda=f=2)$')
                 ]


fig, axes = plt.subplots(2, 3, figsize=(15, 9))

for ax, (dist, continuous, dist_name) in zip(axes.flatten(), distributions):
    if continuous:
        x = np.linspace(dist.ppf(0.0001),
                        dist.ppf(0.9999),
                        300
                        )
        if 'lognorm' in dist.dist.name:
            pdf = stats.lognorm.pdf(x, s=0.7, scale=mean)
        else:
            pdf = dist.pdf(x)
            integrand = integrate.quad(dist.pdf, dist.ppf(0), dist.ppf(1))
            print(integrand)
        ax.plot(x, pdf, 'b-')
        ax.fill_between(x, np.zeros_like(pdf), pdf, alpha=0.3, color='C0')
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

plt.tight_layout(pad=2.5, h_pad=2.0)
fig.savefig('all_likelihoods.pdf')

distributions = [(stats.norm, 2.0, True, 'Gaussian $N(y=3|\mu=f, \sigma=2)$', 3.0),
                 (stats.lognorm, 0.7, True, 'Log-Gaussian $LG(y=3|\mu=f, \sigma=0.7)$', 3.0),
                 (stats.t, 2.0, True, 'Student-T $t(y=3|\mu=f, \sigma=3, df=4)$', 3.0),
                 (stats.beta, 1.6, True, 'Beta $Be(y=0.3|a=f, b=1.6)$', 0.3),
                 (stats.bernoulli, np.nan, False, 'Bernoulli $B(y=1|p=f)$', 1.0),
                 (stats.poisson, np.nan, False, 'Poisson $P(y=3.0|\lambda=f)$', 3.0)
                 ]

fig, axes = plt.subplots(2, 3, figsize=(15, 9))

for ax, (dist, scale, continuous, dist_name, y) in zip(axes.flatten(), distributions):
    ax.set_ylabel('p(y={:.0f}|f)'.format(y))
    if dist == stats.norm:
        ff = np.linspace(-4*scale, 8*scale, 500)
        lik_f = np.nan_to_num(np.array([dist(loc=f, scale=scale).pdf(x=y) for f in ff]))

        dist_in_f = lambda f: dist(loc=f, scale=scale).pdf(x=y)
        integrand = integrate.quad(dist_in_f, -50*scale, 50*scale)[0]
        print(integrand)
    elif dist == stats.t:
        ff = np.linspace(-4*scale, 8*scale, 500)
        lik_f = np.nan_to_num(np.array([dist(loc=f, scale=scale, df=4).pdf(x=y) for f in ff]))

        dist_in_f = lambda f: dist(loc=f, scale=scale, df=4).pdf(x=y)
        integrand = integrate.quad(dist_in_f, -50*scale, 50*scale)[0]
        print(integrand)
    elif dist == stats.lognorm:
        ff = np.linspace(-4*scale, 35*scale, 500)
        lik_f = np.nan_to_num(np.array([dist.pdf(x=y, s=scale, scale=f) for f in ff]))

        dist_in_f = lambda f: dist(scale=f, s=scale).pdf(x=y)
        integrand = integrate.quad(dist_in_f, 1e-15, 35*scale)[0]
        print(integrand)
    elif dist == stats.beta:
        ff = np.linspace(-4*scale, 8*scale, 500)
        lik_f = np.nan_to_num(np.array([dist(a=f, b=scale).pdf(x=y) for f in ff]))
        ax.set_ylabel('p(y={}|f)'.format(y))

        dist_in_f = lambda f: dist(a=f, b=scale).pdf(x=y)
        integrand = integrate.quad(dist_in_f, 1e-15, 35*scale)[0]
        print(integrand)
    elif dist==stats.bernoulli:
        ff = np.linspace(-0.5, 1.5, 500)
        lik_f = np.array([dist(p=f).pmf(y) for f in ff])

        dist_in_f = lambda f: dist(p=f).pmf(y)
        integrand = integrate.quad(dist_in_f, 0, 1)[0]
        print(integrand)
    elif dist==stats.poisson:
        ff = np.linspace(-12, 12, 500)
        lik_f = np.array([dist(f).pmf(y) for f in ff])

        dist_in_f = lambda f: dist(f).pmf(y)
        integrand = integrate.quad(dist_in_f, 0, 100)[0]
        print(integrand)

    ax.plot(ff, lik_f, 'b-')
    ax.fill_between(ff, np.zeros_like(lik_f), lik_f, alpha=0.3, color='C0')
    ax.grid(False)
    ax.title.set_text(dist_name)
    sns.despine()
    ax.set_ylim(bottom=0.0)
    ax.set_xlabel('f')

plt.tight_layout(pad=2.5, h_pad=2.0)
fig.savefig('all_actual_likelihoods.pdf')

