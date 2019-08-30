import numpy as np
from Saul.loglogistic import LogLogistic
import matplotlib.pyplot as plt
import GPy
import seaborn
plt.close('all')
seaborn.set_palette('colorblind')
seaborn.despine()
seaborn.set_style("white")
seaborn.set_context("talk")

location = '../diagrams/'
animation_name = '{}laplace-anim-log'.format(location)
form = 'pdf'
fig_i = 0
lw = 3.0

fig, ax1 = plt.subplots(1,1)
#Note this is how the posterior changes in f, not link_f, thus likelihood has
#full support over continuous X space, as the f will have the link function applied
#to enforce positiveness for example
ax1.set_xlabel('$f_1$')

X = np.zeros((1,1))
k = GPy.kern.RBF(1, variance=2.0)
loglogistic = LogLogistic(r=15.)
bernoulli = GPy.likelihoods.Bernoulli()
poisson = GPy.likelihoods.Poisson()

l = poisson
k.fix()
l.fix()

def g_pdf(x, mu, var):
    return np.exp( -0.5*np.log(2*np.pi) - 0.5*np.log(var) -np.square(x-mu)/var/2)

def generate_save_func(fig):
    fig_i = [0]
    def closure():
        fig.savefig("{}-{}.{}".format(animation_name, fig_i[0], form))
        fig_i[0] += 1
    return closure

save_anim_fig = generate_save_func(fig)

y = 4.0
meta = {'censored':np.ones((1,1))*1}

ff = np.linspace(-8,8,5000).reshape(-1,1)

def nan_to_ninf(x):
    where_are_NaNs = np.isnan(x)
    x[where_are_NaNs] = -np.inf
    return x

#Make and save the prior
#Plot the prior distribution
prior_f = g_pdf(ff,0,k.variance)
log_prior_f = np.log(prior_f)
ax1.plot(ff, log_prior_f, label='log prior, $\log p(f)$', lw=lw)
ax1.legend()
ax1.set_ylim(log_prior_f.min(), log_prior_f.max()+3)
save_anim_fig()

lik_f = np.nan_to_num(np.array([l.pdf(np.atleast_2d(f), np.ones((1,1))*y, meta) for f in ff])[:,:,0])
lik_f = nan_to_ninf(lik_f)
log_lik_f = nan_to_ninf(np.log(lik_f))
#Plot the likelihood
ax1.plot(ff, log_lik_f, label='log likelihood, $\log p(y=4|f)$', lw=lw)
ax1.legend()
ax1.set_ylim(log_prior_f.min(), log_prior_f.max()+3)
save_anim_fig()

#Compute and normalize the posterior. Make sure it goes to 0 both ends
post = prior_f * lik_f
log_post = log_prior_f + log_lik_f
post_normaliser = post.sum()*(ff[1]-ff[0])
post /= post_normaliser
log_post = log_post - np.log(post_normaliser)
# post = nan_to_ninf(np.log(g_pdf(ff.flatten(),0,k.variance))) + nan_to_ninf(np.log(lik_f.flatten()))
# post /= post.sum()*(ff[1]-ff[0])
# post = nan_to_ninf(np.log(post))
#plot the posterior
ax1.plot(ff, log_post, '--', lw=lw, label='log posterior, $\log p(f|y=4)$')
ax1.fill_between(ff.flatten(), log_prior_f.min(), log_post.flatten(), alpha=0.3)
ax1.legend()
ax1.set_ylim(log_prior_f.min(), log_prior_f.max()+3)
save_anim_fig()

#Do the mode finding animation
num_arrows = 5
last_ind = np.argmax(log_post)
first_point = int(ff.shape[0]*0.7)
diff = int(np.abs(first_point - last_ind) / num_arrows)
first_point = last_ind + num_arrows*diff
i = first_point
ap = dict(facecolor='black', frac=0.3, shrink=0.05)
cur_x_point = ff[i,0]
cur_y_point = log_post[i]
arrows = []
points = []
new_point, = plt.plot(cur_x_point, cur_y_point, 'ro', ms=6.0)
points.append(new_point)
save_anim_fig()
for p in range(num_arrows):
    next_x_point = ff[i-diff,0]
    next_y_point = log_post[i-diff]
    new_arr = plt.annotate("", xytext=(cur_x_point, cur_y_point), xy=(next_x_point, next_y_point),
                 xycoords='data', textcoords='data', arrowprops=ap)
    new_point, = plt.plot(next_x_point, next_y_point, 'ro', ms=6.0)
    arrows.append(new_arr)
    points.append(new_point)
    cur_x_point = next_x_point
    cur_y_point = next_y_point
    i = i - diff
    save_anim_fig()

for arr in arrows:
    arr.remove()
for point in points[:-1]:
    point.remove()

if log_post.max() == next_y_point:
    plt.annotate('Evaluate curvature', xy=(next_x_point-2.5, next_y_point+0.5), xycoords='data', textcoords='data')
    save_anim_fig()

#Make some approximations and plot them
#Laplace
from GPy.inference.latent_function_inference import Laplace
m_lap = GPy.core.GP(X=X, Y=np.atleast_2d(y), kernel=k, likelihood=l, Y_metadata=meta, inference_method=Laplace())
lap_m, lap_v = m_lap._raw_predict(X)
ax1.plot(ff, np.log(g_pdf(ff, lap_m, lap_v)), label='', lw=lw)
ax1.vlines(m_lap.inference_method.f_hat, log_prior_f.min(), log_prior_f.max()+3, label='mode, $\hat{f}$', lw=2, alpha=0.5)
plt.legend()
save_anim_fig()

fig2, ax2 = plt.subplots(1,1)
ax2.set_xlabel('$f_1$')
ax2.plot(ff, prior_f, label='prior, $p(f)$', lw=lw)
ax2.plot(ff, lik_f, label='likelihood, $p(y=4|f)$', lw=lw)
ax2.plot(ff, post, '--', lw=lw, label='posterior, $p(f|y=4)$')
ax2.plot(ff, g_pdf(ff, lap_m, lap_v), label='laplace, $q(f)$', lw=lw)
ax2.fill_between(ff.flatten(), prior_f.min(), post.flatten(), alpha=0.3)
ax2.set_ylim(post.min(), post.max()+post.max()*0.2)
ax2.vlines(m_lap.inference_method.f_hat, post.min(), post.max()+post.max()*0.2, label='mode, $\hat{f}$', lw=2, alpha=0.5)
ax2.legend()
animation_name = '{}laplace-anim-full'.format(location)
save_anim_fig2 = generate_save_func(fig2)
save_anim_fig2()
# #VB
# m_vb = GPy.models.GPVariationalGaussianApproximation(X, np.atleast_2d(y), k, l, Y_metadata=meta)
# m_vb.optimize('scg')
# m_vb.optimize('bfgs')
# vb_m, vb_v = m_vb._raw_predict(X)
# plt.plot(ff, g_pdf(ff, vb_m, vb_v), label='vb')

# #ep
# from GPy.inference.latent_function_inference import EP
# m_ep = GPy.core.GP(X=X, Y=np.atleast_2d(y), kernel=k, likelihood=l, Y_metadata=meta, inference_method=EP())
# ep_m, ep_v = m_ep._raw_predict(X)
# plt.plot(ff, g_pdf(ff, ep_m, ep_v), label='ep')
