import numpy as np
import scipy as sp
import GPy
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica'], 'size': 22})
rc('text', usetex=True)
rc('xtick',**{'labelsize':15})
rc('ytick',**{'labelsize':15})

label_fontsize = 22
# matplotlib.rcParams['figure.figsize'] = (10, 6)
matplotlib.rcParams['lines.linewidth'] = 3
plt.close('all')
import seaborn
# seaborn.despine()
# seaborn.set_style("white")
# seaborn.set_context("talk")
form = 'pdf'
title_size = 23
label_size = 20

meta = None

#These are the points on the X axis where we want to look at the correlation between
X_f1 = np.atleast_2d(3.0)
X_f2 = np.atleast_2d(3.4)
X_f3 = np.atleast_2d(4.0)
X_f = np.hstack((X_f1,X_f2,X_f3)).T

marg_mean = np.vstack((X_f1,X_f2,X_f3)).flatten()

#This defines our prior for p(f), the larger the variance, the larger the variance of
#each marginal
variance = 5.0
kern = GPy.kern.RBF(1, variance=variance, lengthscale=1.0)
#These are the correlations between all the points and themselves
fs_cov = kern.K(X_f)
fs_mean = np.zeros(fs_cov.shape[0])

def multivariate_normal_pdf(X, mu, cov):
    N,D = X.shape
    #if mu is None:
    #    mu = np.zeros(D)
    Si, L, Li, logdet = GPy.util.linalg.pdinv(cov)
    X_ = X-mu
    mah = np.sum(np.dot(X_,Si)*X_,1)
    return np.exp(-0.5*np.log(2*np.pi) - 0.5*logdet -0.5*mah)

def inv_logit(f):
    return np.exp(f) / (1 + np.exp(f))

def inv_probit(f, sharpness_param=3.5):
    return 0.5*(1+sp.special.erf((f*sharpness_param)/np.sqrt(2)))

def class_likelihood(y, f):
    return (inv_probit(f)**y) * (1 - inv_probit(f))**(1-y)

res = 300
def make_prior(kern, fs_mean, fs_cov, X, ind_1=0, ind_2=1):
    #Just define the points in which we want to evaluate the prior over
    #(anywhere there is density)
    f1s_min = float(fs_mean[ind_1] - 3*np.sqrt(fs_cov[ind_1,ind_1]))
    f1s_max = float(fs_mean[ind_1] + 3*np.sqrt(fs_cov[ind_1,ind_1]))
    f2s_min = float(fs_mean[ind_2] - 3*np.sqrt(fs_cov[ind_2,ind_2]))
    f2s_max = float(fs_mean[ind_2] + 3*np.sqrt(fs_cov[ind_2,ind_2]))
    f1s_range = np.linspace(f1s_min, f1s_max, res)
    f2s_range = np.linspace(f2s_min, f2s_max, res)
    f12_x, f12_y = np.meshgrid(f1s_range, f2s_range)

    #f1 and f2 locations (possible X1's and X2's) in tuple form
    f12 = np.vstack((f12_x.flatten(), f12_y.flatten())).T

    #Zero mean
    f12_mean = np.zeros_like(fs_mean[:2])

    #Covariance is just covariance between f1 and f2 under our kernel
    ff12 = np.vstack((X[ind_1], X[ind_2]))
    f12_cov = kern.K(ff12)
    f12_cov += np.eye(f12_cov.shape[0])*1e-5

    #Compute the probably density of each X1 and X2 pairing pairing
    f12_prior = multivariate_normal_pdf(f12, f12_mean, f12_cov).reshape(res,res)
    return f12, f12_x, f12_y, f12_prior, f12_cov, f1s_min, f2s_min, f1s_max, f2s_max

#Redefine prior to make it wide to increase sharpness of likelihood
# kern = GPy.kern.RBF(1, variance=80.0)
kern = GPy.kern.RBF(1, variance=10.0)
fs_cov = kern.K(X_f)
fs_mean = np.zeros(fs_cov.shape[0])
f12, f12_x, f12_y, f12_prior, f12_cov, f1s_min, f2s_min, f1s_max, f2s_max = make_prior(kern, fs_mean, fs_cov, X_f, ind_1=0, ind_2=1)

Y = np.ones_like(f12)
like_f12 = (class_likelihood(Y[:,0], f12[:,0]) * class_likelihood(Y[:,1], f12[:,1]))
like_f12 = like_f12.reshape(res,res)

#Unnormalised posterior = prior * likelihood
post_f12 = f12_prior * like_f12

def compute_marginals(unnorm_joint_density, f12_x, normalise=True):
    norm_joint = unnorm_joint_density
    if normalise:
        norm_joint /= unnorm_joint_density.sum()
        norm_joint /= np.diff(f12_x,axis=1)[0,0]**2

    marg1 = unnorm_joint_density.sum(0)
    if normalise:
        marg1 /= marg1.sum()
        marg1 /= (f12_x[0][1] - f12_x[0][0])

    marg2 = unnorm_joint_density.sum(1)
    if normalise:
        marg2 /= marg2.sum()
        marg2 /= (f12_x[0][1] - f12_x[0][0])
    return norm_joint, marg1, marg2

#Renormalize all (Laplace should be normalized already, and EP should be aswell)
post_approx, post_marg1, post_marg2 = compute_marginals(post_f12, f12_x)
lik_approx, lik_marg1, lik_marg2 = like_f12, like_f12[:, -1], like_f12[-1, :]
# lik_approx, lik_marg1, lik_marg2 = compute_marginals(like_f12, f12_x, normalise=False)
prior_approx, prior_marg1, prior_marg2 = compute_marginals(f12_prior, f12_x)

#Plot the two approximations as contours
plt.figure()
plt.contour(f12_x, f12_y, post_approx)
plt.colorbar()
plt.title('Real posterior $p(f|y=1)$')
plt.xlabel('f1', fontsize=label_size)
plt.ylabel('f2', fontsize=label_size)

post_color = 'k'
approx_color= '#CC6600'

#Plot Real posterior and marginals with 3d
ax = plt.subplot2grid((4,3),(1,0), colspan=3, rowspan=3, projection='3d')
contours = np.linspace(0, post_approx.max(), 10)[1:]
ax.contour(f12_x, f12_y, post_approx, contours, colors=post_color)
ax.plot(f12_x[1], post_marg1, f1s_min, zdir='x', color=post_color)
ax.plot(f12_x[1], post_marg2, f2s_max, zdir='y', color=post_color)
ax.set_zlim([0,post_f12.max()])

# import seaborn
from matplotlib import gridspec
# seaborn.set_style('white')

def plot_joint(density, marg1, marg2, f12_x, f12_y, f1s_min, f1s_max, f2s_min, f2s_max, marginals=True):
    #Define grid for subplots
    if marginals:
        gs = gridspec.GridSpec(2, 2, width_ratios=[3, 1], height_ratios=[1, 4])

        fig = plt.figure()
        ax2 = plt.subplot(gs[1,0])
    else:
        fig, ax2 = plt.subplots(1,1)

    cax = ax2.contourf(f12_x, f12_y, density, origin = 'lower',
                       extent=(f1s_min, f1s_max, f2s_min, f2s_max),
                       aspect='auto', cmap=plt.cm.coolwarm, levels=10)
    # ax2.contour(density, levels=np.linspace(density.min()+0.1, density.max(), 5),
                # origin = 'lower', extent = (f1s_min, f1s_max, f2s_min, f2s_max),
                # aspect = 'auto', cmap = plt.cm.bone)	# Contour Lines
    ax2.set_xlabel('$f_1$', fontsize=label_size)
    ax2.set_ylabel('$f_2$', fontsize=label_size)

    if marginals:
        #Turn off all axes
        # ax2.axis('off')

        #Create Y-marginal (right)
        axr = plt.subplot(gs[1,1], sharey=ax2, frameon=False, # xticks=[], yticks=[],
                        xlim=(0, 1.4*marg2.max()), ylim=(f2s_min, f2s_max))
        axr.plot(marg2, f12_x[1], color='black', lw=1)
        axr.fill_betweenx(f12_x[1], 0, marg2, alpha=.75, color='#5673E0')
        axr.xaxis.set_visible(False)
        axr.yaxis.set_visible(False)

        #Create X-marginal (top)
        axt = plt.subplot(gs[0,0], sharex = ax2, frameon = False, #xticks=[], yticks=[],
                        xlim=(f1s_min, f1s_max), ylim=(0, 1.4*marg1.max()))
        axt.plot(f12_x[1], marg1, color='black', lw=1)
        axt.fill_between(f12_x[1], 0, marg1, alpha=.75, color='#5673E0')
        axt.xaxis.set_visible(False)
        axt.yaxis.set_visible(False)

    #Bring the marginals closer to the contour plot
    fig.tight_layout(pad=1)
    return fig

def plot_joint_samples(density, f12_x, f12_y, f1s_min, f1s_max, f2s_min, f2s_max):
    #Define grid for subplots
    def multidim_cumsum(a):
        out = a[::-1,:].cumsum(-1)[::-1,...]
        for i in range(2,a.ndim+1):
            np.cumsum(out, axis=-i, out=out)
        return out
    cdf = multidim_cumsum(density)


    # gibbs sample
    num_samples = 150
    x, y = np.unravel_index(np.argmax(density, axis=None), density.shape)
    samples_x, samples_y = np.zeros(num_samples), np.zeros(num_samples)
    j = 0
    for i in range(num_samples):
        density_x = density[x, :] / density[x, :].sum()
        cdf = np.cumsum(density_x)
        rand_u = np.random.uniform(0,1,1)
        x_new = np.argmin(np.abs(cdf - rand_u))
        if (x_new != 0) and (x_new != density.shape[0]-1):
            x = x_new

        density_y = density[:, y] / density[:, y].sum()
        cdf = np.cumsum(density_y)
        rand_u = np.random.uniform(0,1,1)
        y_new = np.argmin(np.abs(cdf - rand_u))
        if (y_new != 0) and (y_new != density.shape[0]-1):
            y = y_new

        samples_x[j] = x
        samples_y[j] = y
        j += 1

    remove_mask = density[samples_x.astype(int), samples_y.astype(int)] > 5e-3
    order = density[samples_x.astype(int), samples_y.astype(int)]
    samples_x = samples_x[remove_mask]
    samples_y = samples_y[remove_mask]
    samples_x_y = np.array([[x, y] for _, x, y in sorted(zip(order, samples_x, samples_y),
                                                         key=lambda pair: pair[0])])
    sample_x = samples_x_y[:, 0][::-1]
    sample_y = samples_x_y[:, 1][::-1]

    fig, ax = plt.subplots(1,1)
    fake_density = np.ones_like(density)*density.min()
    fake_density[0,0] = density.max()
    cax = ax.contourf(f12_x, f12_y, fake_density, origin = 'lower',
                       extent=(f1s_min, f1s_max, f2s_min, f2s_max),
                       aspect='auto', cmap=plt.cm.coolwarm, levels=10)

    colors = plt.get_cmap('coolwarm')(np.linspace(0, 1, samples_x.shape[0]))
    for x, y, c in zip(f12_x[0, samples_x.astype(int)],
                       f12_y[samples_y.astype(int), 0],
                       colors):
        ax.scatter(x, y, color='C2', alpha=0.5)
    # ax.contour(density, levels=np.linspace(density.min()+0.1, density.max(), 5),
                # origin = 'lower', extent = (f1s_min, f1s_max, f2s_min, f2s_max),
                # aspect = 'auto', cmap = plt.cm.bone)	# Contour Lines
    ax.set_xlabel('$f_1$', fontsize=label_size)
    ax.set_ylabel('$f_2$', fontsize=label_size)

    #Bring the marginals closer to the contour plot
    fig.tight_layout()#pad=1)
    return fig

fig_joint_like = plot_joint(lik_approx, lik_marg1, lik_marg2, f12_x, f12_y, f1s_min, f1s_max, f2s_min, f2s_max)
fig_joint_prior = plot_joint(prior_approx, prior_marg1, prior_marg2, f12_x, f12_y, f1s_min, f1s_max, f2s_min, f2s_max)

fig_joint_post = plot_joint(post_approx, post_marg1, post_marg2, f12_x, f12_y, f1s_min, f1s_max, f2s_min, f2s_max)

fig_joint_post_samples = plot_joint_samples(post_approx, f12_x, f12_y, f1s_min, f1s_max, f2s_min, f2s_max)

fig_joint_like.suptitle('Likelihood $p(y_{1}=1, y_{2}=1|f_1,f_2)$', fontsize=title_size)
fig_joint_prior.suptitle('Prior $p(f_1,f_2)$', fontsize=title_size)
fig_joint_post.suptitle('True posterior $p(f_1,f_2|y_{1}=1, y_{2}=1)$', fontsize=title_size)
# fig_joint_post_samples.suptitle('Samples of posterior $p(f_1,f_2|y_{1}=1, y_{2}=1)$', fontsize=title_size)

fig_margs, ax_margs = plt.subplots(1,1)
approxs = [('posterior $p(f|y=1)$', post_marg1, 'C0'),
           ('likelihood $p(y=1|f)$', lik_marg1, 'C1'),
           ('prior $p(f)$', prior_marg1, 'C2')]
for (name, marg, color) in approxs:
    ax_margs.plot(f12_x[0], marg, color=color, label=name, lw=2.0)
    if name.startswith('posterior'):
        ax_margs.fill_between(f12_x[0], 0, marg, color=color, alpha=.5)

leg = ax_margs.legend(fontsize=label_fontsize/1.5, frameon=True, bbox_to_anchor=(0.52, 1.00))
leg.get_frame().set_linewidth(0.0)
# ax_margs.legend()
ax_margs.set_ylim(bottom=0.0)
ax_margs.set_xlabel("$f_{1}$")
fig_margs.tight_layout()
#fig_margs.suptitle('Marginals for $f_1$', fontsize=title_size)

fig_joint_like.savefig('2d_joint_likelihood.pdf')
fig_joint_prior.savefig('2d_joint_prior.pdf')
fig_joint_post.savefig('2d_joint_posterior.pdf')
fig_joint_post_samples.savefig('2d_joint_posterior_samples.pdf')
fig_margs.savefig('1d_marginals.pdf')

import numpy as np
import scipy as sp
import GPy
import matplotlib.pyplot as plt
save=True
# sns.set_style("white")
# sns.set_context("paper")
# sns.set_palette('Set2')

label_fontsize = 22
#import prettyplotlib as pp
fig_dir='.'

# Make the GP draws
def rbf(X1, X2):
    return np.exp(-0.5*((X1-X2.T)**2))
# old_f_locations = X.copy()
# X_f1 = np.atleast_2d(3)
# X_f2 = np.atleast_2d(4)
# X_f = np.hstack((X_f1,X_f2,X_f3)).T

X_before_f1 = np.linspace(X_f.min()-2, X_f1[0,0], 50)[:, None]
X_between_f1_f2 = np.linspace(X_f1[0,0], X_f2[0,0], 50)[:, None]
X_after_f2 = np.linspace(X_f2[0,0], X_f.max()+2, 50)[:, None]
X = np.vstack([X_before_f1, X_between_f1_f2, X_after_f2])
K = kern.K(X)
#Plot prior
F = np.random.multivariate_normal(np.zeros(X.shape[0]), K, 50)
squashed_F = inv_probit(F.T)

def plot_prior(Xs):
    fig_prior, ax = plt.subplots(1, 1, figsize=(10, 5), sharex=True)
    ax.plot(X, F.T, 'C0', lw=1)
    for i, X_ in enumerate(Xs):
        ax.axvline(X_, label='$f_{}$'.format(i+1), c='k')
        # ax.axvline(X_f[1], F.min(), F.max(), label='$f_2$', c='k')
    leg = plt.legend(fontsize=label_fontsize/1.5, frameon=True, bbox_to_anchor=(1.02, 1.00))
    leg.get_frame().set_linewidth(0.0)
    seaborn.despine()
    ax.set_ylabel('$f$', fontsize=label_fontsize)
    ax.set_xlim(X.min(), X.max())
    return fig_prior

def plot_prior_squashed(Xs):
    fig_squashed, ax_squashed = plt.subplots(1, 1, figsize=(10, 5), sharex=True)
    ax_squashed.plot(X, squashed_F, 'C0', lw=1)
    for i, X_ in enumerate(Xs):
        ax_squashed.axvline(X_, label='$f_{}$'.format(i+1), c='k')
    leg = plt.legend(fontsize=label_fontsize/1.5, frameon=True, bbox_to_anchor=(1.02, 1.00))
    leg.get_frame().set_linewidth(0.0)
    ax_squashed.set_ylabel('$\lambda(f)$', fontsize=label_fontsize)
    ax_squashed.set_xlabel('$x$', fontsize=label_fontsize)
    seaborn.despine()
    ax_squashed.set_ylim(0, 1)
    ax_squashed.set_xlim(X.min(), X.max())
    return fig_squashed

def plot_likelihood_alpha(Xs):
    all_lik = np.ones((F.shape[0], 1))

    for X_ in Xs:
        obs_X = X_
        obs_y = np.ones_like(X_)
        X_ind = np.argmax(X == X_)
        f_ = F.T[X_ind]
        lik_ = class_likelihood(obs_y, f_[:, None])
        all_lik *= lik_

    all_lik = ((all_lik- all_lik.min()) / (all_lik.max() - all_lik.min()))

    # Hack the gradient
    # lik_f1[0, np.argmin(lik_f1)] += 0.00001
    # lik_f1[0, np.argmax(lik_f1)] -= 0.00001
    #lik_f1 *= 5.0

    fig_lik, ax = plt.subplots(1, 1, figsize=(10, 5))

    for i in range(all_lik.shape[0]):
        ax.plot(X, squashed_F[:,i:i+1], 'C0', lw=1, alpha=all_lik[i, 0])

    for i, X_ in enumerate(Xs):
        ax.axvline(X_, label='$f_{}$'.format(i+1), c='k')
        ax.plot(X_, 1, color='C1', marker='o', markersize=12)
    leg = plt.legend(fontsize=label_fontsize/1.5, frameon=True, bbox_to_anchor=(1.02, 1.00))
    leg.get_frame().set_linewidth(0.0)
    seaborn.despine()
    ax.set_ylabel('$\lambda(f)$', fontsize=label_fontsize)
    ax.set_xlabel('$x$', fontsize=label_fontsize)
    ax.set_xlim(X.min(), X.max())
    return fig_lik

fig_1d_prior = plot_prior(Xs=[X_f[0]])
fig_1d_prior_squashed = plot_prior_squashed(Xs=[X_f[0]])
fig_lik_1d = plot_likelihood_alpha(Xs=[X_f[0]])

if save:
    fig_1d_prior.savefig('{}/1d_gp_prior_samples.pdf'.format(fig_dir), bbox_inches='tight')
    fig_1d_prior_squashed.savefig('{}/1d_squashed_gp_prior_samples.pdf'.format(fig_dir), bbox_inches='tight')
    fig_lik_1d.savefig('{}/1d_gp_post_samples.pdf'.format(fig_dir), bbox_inches='tight')

fig_2d_prior = plot_prior(Xs=[X_f[0], X_f[1]])
fig_2d_prior_squashed = plot_prior_squashed(Xs=[X_f[0], X_f[1]])
fig_lik_2d = plot_likelihood_alpha(Xs=[X_f[0], X_f[1]])

if save:
    fig_2d_prior.savefig('{}/2d_gp_prior_samples.pdf'.format(fig_dir), bbox_inches='tight')
    fig_2d_prior_squashed.savefig('{}/2d_squashed_gp_prior_samples.pdf'.format(fig_dir), bbox_inches='tight')
    fig_lik_2d.savefig('{}/2d_gp_post_samples.pdf'.format(fig_dir), bbox_inches='tight')


"""
X_real_loc = np.array([2.5, 4])[:, None]
X = np.sin(X_real_loc)
K = rbf(X, X)
X_test_loc = np.linspace(X_f.min()-2, X_f.max()+2, 100)[:, None]
K = rbf(X_real_loc, X_real_loc)
Ks = rbf(X_real_loc, X_test_loc)
Kss = rbf(X_test_loc, X_test_loc)
Ki = np.linalg.inv(K)
mean = np.dot(Ks.T, np.dot(Ki, X))
mean = np.squeeze(mean)
cov = Kss - np.dot(Ks.T, np.dot(Ki, Ks))
Fs = np.random.multivariate_normal(mean, cov, 10)
plt.figure(2)
#plt.hold(True)
plt.scatter(X_real_loc, X, marker='D', linewidths=5)
plt.plot(X_test_loc, Fs.T)
plt.axvline(X_f[0], F.min(), F.max(), label='$f_1$', c='g')
plt.axvline(X_f[1], F.min(), F.max(), label='$f_2$', c='r')

leg = plt.legend(fontsize=label_fontsize/1.5, frameon=True, bbox_to_anchor=(1.02, 1.00))
leg.get_frame().set_linewidth(0.0)

plt.axis([X_f.min()-2, X_f.max()+2, -3, 3])
plt.ylabel('$f$', fontsize=label_fontsize)
plt.xlabel('$x$', fontsize=label_fontsize)
sns.despine()
if save:
    plt.savefig('{}/gp_fit.pdf'.format(fig_dir), bbox_inches='tight')
"""
