import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn
seaborn.despine()
seaborn.set_style("white")
seaborn.set_context("talk")
form = 'pdf'

location = '../diagrams/'

def generate_save_func(fig, animation_name):
    fig_i = [0]
    def closure():
        fig.savefig("{}-{}.{}".format(animation_name, fig_i[0], form))
        fig_i[0] += 1
    return closure

def KLD(m1, v1, m2, v2):
    return np.log(np.sqrt(v2)) - np.log(np.sqrt(v1)) + (v1 + (m1 - m2)**2)/(2.0*v2) - 0.5

def plot_mu(m1, v1, m2, v2, fileprefix=''):
    KL = np.zeros_like(m2)
    X = np.linspace(-6.0, 6.0, 100)
    pdf = stats.norm.pdf(X, loc=m1, scale=np.sqrt(v1))
    fig1, (ax1, ax2) = plt.subplots(2,1)
    animation_name = "KL_gaussian_mu"
    save_mu_anim_fig = generate_save_func(fig1, animation_name=animation_name)
    lines = []
    for i, mean in enumerate(m2):
        KL[i] = KLD(m1,v1,mean,v2)
        pdfi = stats.norm.pdf(X, loc=mean, scale=np.sqrt(v2))
        ri, = ax1.plot(X, pdfi, 'r-')
        bi, = ax1.plot(X, pdf, 'b--')
        ax1.set_ylabel('pdf')
        ax1.set_xlabel('x')
        ax2.plot(m2[:i+1], KL[:i+1], ls='-', marker='.', color='r', markersize=15)
        ax2.set_xlabel('$\mu$')
        ax2.set_xlim(m2.min(), m2.max())
        ax2.set_ylim(0, KL.max())
        ax2.set_ylabel('KL$[q(x)||p(x)]$')
        ax1.legend([ri, bi], ["q(x) ~ $N(\\mu={}, \\sigma^2={})$".format(mean, v2), "p(x) ~ $N({}, {})$".format(m1, v1)])
        #Make old lines fade out
        if len(lines) > 0:
            lines[-1].set_alpha(0.1)
        lines.append(ri)
        save_mu_anim_fig()

def plot_var(m1, v1, m2, v2, fileprefix=''):
    KL = np.zeros_like(v2)
    X = np.linspace(-6.0, 6.0, 100)
    pdf = stats.norm.pdf(X, loc=m1, scale=np.sqrt(v1))
    fig1, (ax1, ax2) = plt.subplots(2,1)
    animation_name = "KL_gaussian_var"
    save_var_anim_fig = generate_save_func(fig1, animation_name=animation_name)
    lines = []
    for i, var in enumerate(v2):
        KL[i] = KLD(m1,v1,m2,var)
        pdfi = stats.norm.pdf(X, loc=m1, scale=np.sqrt(var))
        ri, = ax1.plot(X, pdfi, 'r-', label='q(x)')
        bi, = ax1.plot(X, pdf, 'b--', label='p(x)')
        ax1.set_ylabel('pdf')
        ax1.set_xlabel('x')
        ax2.plot(v2[:i+1], KL[:i+1], ls='-', marker='.', color='r', markersize=15)
        ax2.set_xlabel('$\sigma^2$')
        ax2.set_xlim(v2.min(), v2.max())
        ax2.set_ylim(0, KL.max())
        ax2.set_ylabel('KL$[q(x)||p(x)]$')
        ax1.legend([ri, bi], ["q(x) ~ $N(\\mu={}, \\sigma^2={})$".format(m2, var), "p(x) ~ $N({}, {})$".format(m1, v1)])
        #Make old lines fade out
        if len(lines) > 0:
            lines[-1].set_alpha(0.1)
        lines.append(ri)
        save_var_anim_fig()

m1 = 0.0
v1 = 1.0

m2 = np.linspace(-2,2,5)
v2 = 1.0
plot_mu(m1, v1, m2, v2, fileprefix='mu')

m2 = 0.0
v2 = np.linspace(0.3,2,5)
plot_var(m1, v1, m2, v2, fileprefix='var')
plt.show()
