import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import GPy
import seaborn
plt.close('all')
aspect = 0.4
alpha = 1.0
plot_kw = {'bbox_inches':'tight', 'pad_inches':0}
poisson = True
bernoulli = True
gp = True
animate = False

colors = seaborn.color_palette('Set2', 2)

def plot_model_with_samples(m, samples_label='Realizations of $f(x)$'):
    Xdiff = np.abs(m.X.max() - m.X.min())
    Xpred = np.linspace(m.X.min() - Xdiff*0.4, m.X.max() + Xdiff*0.4, 200)[:, None]
    f_samples = m.posterior_samples_f(Xpred)
    f_samples = m.likelihood.gp_link.transf(f_samples)
    fig, ax = plt.subplots(1,1)

    # fig_gp.suptitle('Gaussian process realisations, Gaussian noise')
    ax.plot(Xpred, f_samples[:,0,:-1], c=colors[1], alpha=alpha)
    ax.plot(Xpred, f_samples[:,0,-1], c=colors[1], alpha=alpha, label=samples_label)
    ax.plot(m.X, m.Y, 'ko', label='Observations')

    fmu, fv = m._raw_predict(Xpred)
    lower, upper = m.likelihood.predictive_quantiles(fmu, fv, (2.5, 97.5))
    from GPy.plotting.matplot_dep.base_plots import meanplot
    # from GPy.plotting.matplot_dep import Tango

    linecol='b'#,Tango.colorsHex['darkBlue']
    fillcol='b'#Tango.colorsHex['lightBlue']
    edgecol = linecol
    #here's the box
    kwargs = {}
    kwargs['linewidth']=0.5
    if not 'alpha' in kwargs.keys():
        kwargs['alpha'] = 0.3
    lower = lower[:,0].flatten()
    upper = upper[:,0].flatten()
    x = Xpred.flatten()
    ax.fill(np.hstack((x,x[::-1])),
            np.hstack((upper,lower[::-1])),color=fillcol,label='95% credible intervals for $p(y^*|y)$', **kwargs)


    #this is the edge:
    meanplot(Xpred, upper,color=edgecol,linewidth=0.2,ax=ax)
    meanplot(Xpred, lower,color=edgecol,linewidth=0.2,ax=ax)
    # ax.legend()
    ax.set_xlim(xmin=Xpred.min(), xmax=Xpred.max())

    fig.tight_layout()
    ax.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
            ncol=3, mode="expand", borderaxespad=0.)
    return fig

np.random.seed(1)
#Poisson example
X = np.linspace(0,23.50,48)[:, None]
fs = 3*np.sin(10 + 0.6*X) + np.sin(0.1*X)
intensities = fs + 4
Yp = np.array([sp.random.poisson(intensity) for intensity in intensities])
Y = np.array([sp.random.normal(f, scale=1.0) for f in fs])

if poisson:
    kernel = GPy.kern.RBF(1, variance=1.0, lengthscale=1.0)
    poisson_likelihood = GPy.likelihoods.Poisson()
    laplace_inf = GPy.inference.latent_function_inference.Laplace()
    m_poisson = GPy.core.GP(X=X, Y=Yp, likelihood=poisson_likelihood, inference_method=laplace_inf, kernel=kernel)
    m_poisson.optimize('bfgs', messages=1)
    # fig_poisson, ax_poisson = plt.subplots(1,1)
    # lines = m_poisson.plot(ax=ax_poisson, resolution=400, samples=10, apply_link=True, plot_training_data=False)
    # plt.setp(lines['posterior_samples'], label='Realizations of $\exp(f(x))$')
    # posterior_fill = lines['gpplot'][1]
    # plt.setp(posterior_fill, label='95% credible intervals for $p(y^*|y)$')
    # ax_poisson.plot(m_poisson.X, m_poisson.Y, 'ko', label='Observations')
    # ax_poisson.set_xlabel('x')
    # ax_poisson.set_ylabel('Counts')
    # fig_poisson.tight_layout()
    # ax_poisson.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
            # ncol=3, mode="expand", borderaxespad=0.)
    # ax_poisson.set_aspect(aspect)
    # fig_poisson.savefig('PoissonPosterior.pdf', **plot_kw)

    fig_poisson = plot_model_with_samples(m_poisson, samples_label='Realizations of $\exp(f(x))$')
    fig_poisson.gca().set_aspect(0.55)
    fig_poisson.savefig('Poisson_samples.pdf', **plot_kw)

# default_seed = 10000
default_seed = 1
def toy_linear_1d_classification(seed=default_seed, optimize=True, plot=True, axes=None):
    """
    Simple 1D classification example using EP approximation

    :param seed: seed value for data generation (default is 4).
    :type seed: int

    """

    try:import pods
    except ImportError:print('pods unavailable, see https://github.com/sods/ods for example datasets')
    data = pods.datasets.toy_linear_1d_classification(seed=seed)
    Y = data['Y'][:, 0:1]
    Y[Y.flatten() == -1] = 0

    # Model definition
    m = GPy.models.GPClassification(data['X'], Y)

    # Optimize
    if optimize:
        #m.update_likelihood_approximation()
        # Parameters optimization:
        print("Pre opt")
        print(m)
        m.optimize('bfgs', messages=1)
        print("EP 1 opt")
        print(m)
        m.optimize('bfgs', messages=1)
        print("EP 2 opt")
        print(m)
        m.optimize('bfgs', messages=1)
        print("EP 3 opt")
        print(m)
        #m.update_likelihood_approximation()
        # m.pseudo_EM()

    # Plot
    if plot:
        from matplotlib import pyplot as plt
        if axes is None:
            fig, axes = plt.subplots(2, 1)
        m.plot_f(ax=axes[0])
        m.plot(ax=axes[1])

    print(m)
    return m

def toy_linear_1d_classification_laplace(seed=default_seed, optimize=True, plot=True, axes=None):
    """
    Simple 1D classification example using Laplace approximation

    :param seed: seed value for data generation (default is 4).
    :type seed: int

    """

    try:import pods
    except ImportError:print('pods unavailable, see https://github.com/sods/ods for example datasets')
    data = pods.datasets.toy_linear_1d_classification(seed=seed)
    Y = data['Y'][:, 0:1]
    Y[Y.flatten() == -1] = 0

    likelihood = GPy.likelihoods.Bernoulli()
    laplace_inf = GPy.inference.latent_function_inference.Laplace()
    kernel = GPy.kern.RBF(1)

    # Model definition
    m = GPy.core.GP(data['X'], Y, kernel=kernel, likelihood=likelihood, inference_method=laplace_inf)

    # Optimize
    if optimize:
        try:
            print("Pre opt")
            print(m)
            m.optimize('bfgs', messages=1)
            print("Laplace opt 1")
            print(m)
            m.optimize('bfgs', messages=1)
            print("Laplace opt 2")
            print(m)
            m.optimize('bfgs', messages=1)
            print("Laplace opt 3")
            print(m)
        except Exception as e:
            return m

    # Plot
    if plot:
        from matplotlib import pyplot as plt
        if axes is None:
            fig, axes = plt.subplots(2, 1)
        m.plot_f(ax=axes[0])
        m.plot(ax=axes[1])

    print(m)
    return m

#Bernoulli example
if bernoulli:
    fig_ep, axes_ep = plt.subplots(2, 1)
    m_ep = toy_linear_1d_classification(seed=default_seed, optimize=True, plot=True, axes=axes_ep)
    axes_ep[0].set_title('EP approximation')
    axes_ep[0].set_ylabel('latent function f')
    plt.show()

    fig_laplace, axes_laplace = plt.subplots(2, 1)
    m_lap = toy_linear_1d_classification_laplace(seed=default_seed, optimize=True, plot=True, axes=axes_laplace)
    axes_laplace[0].set_title('Laplace approximation')

    Xdiff = np.abs(m_ep.X.max() - m_ep.X.min())
    Xpred = np.linspace(m_ep.X.min() - Xdiff*0.4, m_ep.X.max() + Xdiff*0.4, 200)[:, None]
    f_samples = m_ep.posterior_samples_f(Xpred)
    link_f_samples = m_ep.likelihood.gp_link.transf(f_samples)

    fig_bern, (ax_bern_f, ax_bern_link_f) = plt.subplots(1,2, figsize=(5,3))
    ax_bern_f.plot(Xpred, f_samples[:,0,:-1], c=colors[0], alpha=alpha)
    ax_bern_f.plot(Xpred, f_samples[:,0,-1], c=colors[0], alpha=alpha, label='Realizations of $f(x)$')
    ax_bern_f.set_xlim(Xpred.min(), Xpred.max())

    ax_bern_link_f.plot(Xpred, link_f_samples[:,0,:-1], c=colors[1], alpha=alpha)
    ax_bern_link_f.plot(Xpred, link_f_samples[:,0,-1], c=colors[1], alpha=alpha, label='Realizations of $\Phi(f(x))$')
    ax_bern_link_f.plot(m_ep.X, m_ep.Y, 'ko', label='Observations')
    ax_bern_link_f.set_xlim(Xpred.min(), Xpred.max())

    # ax_bern_f.set_aspect(0.7)
    # ax_bern_link_f.set_aspect(0.7)
    # fig_bern.suptitle('Classification example')
    ax_bern_f.set(aspect=0.8)
    ax_bern_link_f.set(aspect=18.0)
    fig_bern.tight_layout()
    ax_bern_f.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
            ncol=2, mode="expand", borderaxespad=0.)
    ax_bern_link_f.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
            mode="expand", borderaxespad=0.)
    # fig_bern.gca().set_aspect(0.7)
    fig_bern.savefig('BernoulliSamples.pdf', **plot_kw)

#GP plot
if gp:
    # seaborn.despine()
    # seaborn.set_style("white")
    # seaborn.set_context("talk")
    form = 'pdf'

    m_gp = GPy.models.GPRegression(X, Y)
    m_gp.optimize('bfgs')

    fig_gp = plot_model_with_samples(m_gp)
    fig_gp.gca().set_aspect(0.9)
    fig_gp.savefig('GaussianSamples.pdf', **plot_kw)

    """
    if animate:
        i = 10
        latent_func = 3

        c = 'r'

        #Pick out the latent function
        ax.plot(Xpred, f_samples[:,latent_func], c=colors[1], alpha=alpha, label='Realizations of $f(x)$')

        #Pick out a specific datum
        plt.plot(X[i], Y[i], 'bo')

        def find_nearest(a, a0):
            "Element in nd array `a` closest to the scalar value `a0`"
            idx = np.abs(a - a0).argmin()
            return idx

        Xpred_ind = find_nearest(Xpred, X[i])
        f_val = f_samples[Xpred_ind, latent_func]
        plt.plot(Xpred[Xpred_ind], f_val, 'ro')

        #inset
        gauss_var = float(m_gp.likelihood.variance.values)
        size = 3*np.sqrt(gauss_var)
        pdf_range = np.linspace(f_val - size, f_val + size)
        inset_a = plt.axes([.65, .65, 0.3, 0.3], xticks=[])
        lik_f = sp.stats.norm(loc=f_val, scale = np.sqrt(gauss_var)).pdf(pdf_range)
        inset_a.plot(lik_f, pdf_range, color='black')
        inset_a.axhline(Y[i], 0)
        inset_a.fill_betweenx(pdf_range, 0, lik_f, alpha=.75, color=c)

        # axr = plt.subplot(gs[1,1], sharey=ax2, frameon=False, # xticks=[], yticks=[],
                        # xlim=(0, 1.4*marg2.max()), ylim=(f2s_min, f2s_max))
        # axr.plot(marg2, f12_x[1], color='black')
        # axr.fill_betweenx(f12_x[1], 0, marg2, alpha=.75, color='#5673E0')
        # axr.xaxis.set_visible(False)
        # axr.yaxis.set_visible(False)

    """
