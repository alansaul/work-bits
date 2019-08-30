import numpy as np
import scipy as sp
import GPy
import matplotlib.pyplot as plt
plt.close('all')
import seaborn
# seaborn.despine()
seaborn.set_style("white")
seaborn.set_context("talk")
form = 'pdf'
title_size = 23
label_size = 20

meta = None

#These are the points on the X axis where we want to look at the correlation between
X_f1 = np.atleast_2d(0.1)
X_f2 = np.atleast_2d(0.4)
X_f3 = np.atleast_2d(4)
X_f = np.hstack((X_f1,X_f2,X_f3)).T

marg_mean = np.vstack((X_f1,X_f2,X_f3)).flatten()

#This defines our prior for p(f), the larger the variance, the larger the variance of
#each marginal
variance = 5.0
kern = GPy.kern.RBF(1, variance=variance)
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

def inv_probit(f, sharpness_param=1):
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
kern = GPy.kern.RBF(1, variance=60.0)
fs_cov = kern.K(X_f)
fs_mean = np.zeros(fs_cov.shape[0])
f12, f12_x, f12_y, f12_prior, f12_cov, f1s_min, f2s_min, f1s_max, f2s_max = make_prior(kern, fs_mean, fs_cov, X_f, ind_1=0, ind_2=1)

# #Choose the space on which we want to look at f
# f = np.linspace(-2,2,res)[:, None]
# Y = np.ones_like(f)

# plt.plot(f, class_likelihood(Y,f))
# plt.xlabel('f')
# plt.ylabel('p(Y=1|f)')

# #Now lets look at the chosen points of X_f1 and X_f2 only
# plt.figure()
# plt.contour(f12_x, f12_y, f12_prior)
# plt.colorbar()
# plt.title('Prior p(f)')
# plt.xlabel('f1')
# plt.ylabel('f2')

Y = np.ones_like(f12)
like_f12 = (class_likelihood(Y[:,0], f12[:,0]) * class_likelihood(Y[:,1], f12[:,1]))
like_f12 = like_f12.reshape(res,res)
# plt.figure()
# plt.contour(f12_x, f12_y, like_f12)
# plt.colorbar()
# plt.title('Binomial Likelihood p(y=1|f)')
# plt.xlabel('f1')
# plt.ylabel('f2')

#Appoximate posterior
Y=np.ones_like(X_f)

laplace = GPy.inference.latent_function_inference.Laplace()
likelihood = GPy.likelihoods.bernoulli.Bernoulli(gp_link=GPy.likelihoods.link_functions.Probit())
lap_post, lap_log_marginal, lap_grads = laplace.inference(kern, X_f, likelihood, Y)
lap_mode = lap_post.mean
lap_covar = lap_post.covariance
lap_approx = multivariate_normal_pdf(f12, lap_mode[:2].T, lap_covar[:2, :2]).reshape(res, res)

ep = GPy.inference.latent_function_inference.EP()
ep_post, ep_log_marginal, ep_grads = ep.inference(kern, X_f, likelihood, Y)
ep_mode = ep_post.mean
ep_covar = ep_post.covariance
ep_approx = multivariate_normal_pdf(f12, ep_mode[:2].T, ep_covar[:2, :2]).reshape(res, res)

m_vb = GPy.models.GPVariationalGaussianApproximation(X_f, np.atleast_2d(Y), kernel=kern,
                                                     likelihood=likelihood, Y_metadata=meta)
m_vb.fix()
m_vb.alpha.unfix()
m_vb.beta.unfix()
m_vb.optimize('scg')
m_vb.optimize('bfgs')
kl_post = m_vb.posterior
kl_mode = kl_post.mean
kl_covar = kl_post.covariance
kl_approx = multivariate_normal_pdf(f12, kl_mode[:2].T, kl_covar[:2, :2]).reshape(res, res)

#Unnormalised posterior = prior * likelihood
post_f12 = f12_prior * like_f12

def compute_marginals(unnorm_joint_density, f12_x):
    norm_joint = unnorm_joint_density / unnorm_joint_density.sum()
    norm_joint /= np.diff(f12_x,axis=1)[0,0]**2

    marg1 = unnorm_joint_density.sum(0)
    marg1 /= marg1.sum()
    marg1 /= (f12_x[0][1] - f12_x[0][0])

    marg2 = unnorm_joint_density.sum(1)
    marg2 /= marg2.sum()
    marg2 /= (f12_x[0][1] - f12_x[0][0])
    return norm_joint, marg1, marg2

#Renormalize all (Laplace should be normalized already, and EP should be aswell)
lap_approx, lap_marg1, lap_marg2 = compute_marginals(lap_approx, f12_x)
ep_approx, ep_marg1, ep_marg2 = compute_marginals(ep_approx, f12_x)
kl_approx, kl_marg1, kl_marg2 = compute_marginals(kl_approx, f12_x)
post_approx, post_marg1, post_marg2 = compute_marginals(post_f12, f12_x)

lik_approx, lik_marg1, lik_marg2 = compute_marginals(like_f12, f12_x)
prior_approx, prior_marg1, prior_marg2 = compute_marginals(f12_prior, f12_x)

#Plot the two approximations as contours
plt.figure()
plt.contour(f12_x, f12_y, post_approx)
plt.colorbar()
plt.title('Real posterior p(f|y=1)')
plt.xlabel('f1', fontsize=label_size)
plt.ylabel('f2', fontsize=label_size)

plt.contour(f12_x, f12_y, lap_approx)
plt.colorbar()
plt.title('Laplace approximate posterior p(f|y=1)')
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

contours = np.linspace(0, lap_approx.max(), 10)[1:]
ax.contour(f12_x, f12_y, lap_approx, contours, colors=approx_color)
ax.plot(f12_x[1], lap_marg1, f1s_min, zdir='x', color=approx_color)
ax.plot(f12_x[1], lap_marg2, f2s_max, zdir='y', color=approx_color)

ax.set_zlim([0,post_f12.max()])

# import seaborn
from matplotlib import gridspec
# seaborn.set_style('white')

def plot_joint(density, marg1, marg2, f12_x, f12_y, f1s_min, f1s_max, f2s_min, f2s_max):
    #Define grid for subplots
    gs = gridspec.GridSpec(2, 2, width_ratios=[3, 1], height_ratios=[1, 4])

    fig = plt.figure()
    ax2 = plt.subplot(gs[1,0])
    cax = ax2.contourf(f12_x, f12_y, density, origin = 'lower',
            extent=(f1s_min, f1s_max, f2s_min, f2s_max), aspect='auto', cmap=plt.cm.coolwarm)
    ax2.contour(density, origin = 'lower', extent = (f1s_min, f1s_max, f2s_min, f2s_max),
                aspect = 'auto', cmap = plt.cm.bone)	# Contour Lines
    ax2.set_xlabel('$f_1$', fontsize=label_size)
    ax2.set_ylabel('$f_2$', fontsize=label_size)

    #Turn off all axes
    # ax2.axis('off')

    #Create Y-marginal (right)
    axr = plt.subplot(gs[1,1], sharey=ax2, frameon=False, # xticks=[], yticks=[],
                    xlim=(0, 1.4*marg2.max()), ylim=(f2s_min, f2s_max))
    axr.plot(marg2, f12_x[1], color='black')
    axr.fill_betweenx(f12_x[1], 0, marg2, alpha=.75, color='#5673E0')
    axr.xaxis.set_visible(False)
    axr.yaxis.set_visible(False)

    #Create X-marginal (top)
    axt = plt.subplot(gs[0,0], sharex = ax2, frameon = False, #xticks=[], yticks=[],
                    xlim=(f1s_min, f1s_max), ylim=(0, 1.4*marg1.max()))
    axt.plot(f12_x[1], marg1, color='black')
    axt.fill_between(f12_x[1], 0, marg1, alpha=.75, color='#5673E0')
    axt.xaxis.set_visible(False)
    axt.yaxis.set_visible(False)

    #Bring the marginals closer to the contour plot
    fig.tight_layout(pad=1)
    return fig

fig_joint_like = plot_joint(lik_approx, lik_marg1, lik_marg2, f12_x, f12_y, f1s_min, f1s_max, f2s_min, f2s_max)
fig_joint_prior = plot_joint(prior_approx, prior_marg1, prior_marg2, f12_x, f12_y, f1s_min, f1s_max, f2s_min, f2s_max)

fig_joint_laplace = plot_joint(lap_approx, lap_marg1, lap_marg2, f12_x, f12_y, f1s_min, f1s_max, f2s_min, f2s_max)
fig_joint_post = plot_joint(post_approx, post_marg1, post_marg2, f12_x, f12_y, f1s_min, f1s_max, f2s_min, f2s_max)
fig_joint_ep = plot_joint(ep_approx, ep_marg1, ep_marg2, f12_x, f12_y, f1s_min, f1s_max, f2s_min, f2s_max)
fig_joint_kl = plot_joint(kl_approx, kl_marg1, kl_marg2, f12_x, f12_y, f1s_min, f1s_max, f2s_min, f2s_max)

fig_joint_like.suptitle('Likelihood $p(y=1|f_1,f_2)$', fontsize=title_size)
fig_joint_prior.suptitle('Prior $p(f_1,f_2)$', fontsize=title_size)
fig_joint_post.suptitle('True posterior', fontsize=title_size)
fig_joint_laplace.suptitle('Laplace approximation', fontsize=title_size)
fig_joint_ep.suptitle('EP approximation', fontsize=title_size)
fig_joint_kl.suptitle('KL approximation', fontsize=title_size)

fig_margs, ax_margs = plt.subplots(1,1)
approxs = [('laplace', lap_marg1, 'b'), ('posterior', post_marg1, 'r'), ('EP', ep_marg1, 'g'), ('KL', kl_marg1, 'm')]
for (name, marg, color) in approxs:
    ax_margs.plot(f12_x[1], marg, color=color, label=name)
    if name == 'posterior':
        ax_margs.fill_between(f12_x[1], 0, marg, color=color, alpha=.5, label=name)
ax_margs.legend()
fig_margs.suptitle('Marginals for $f_2$ compared', fontsize=title_size)

fig_joint_like.savefig('joint_likelihood.pdf')
fig_joint_prior.savefig('joint_prior.pdf')
fig_joint_post.savefig('joint_posterior.pdf')
fig_joint_ep.savefig('joint_ep.pdf')
fig_joint_kl.savefig('joint_kl.pdf')
fig_joint_laplace.savefig('joint_laplace.pdf')
fig_margs.savefig('marginals.pdf')
