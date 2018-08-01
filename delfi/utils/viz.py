import delfi.utils.colormaps as cmaps
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import time


def loss(losses, key='trn', loss_clipping=1000., title=''):
    """Given an info dict, plot loss"""

    x = np.array(losses[key + '_iter'])
    y = np.array(losses[key + '_val'])

    clip_idx = np.where(y > loss_clipping)[0]
    if len(clip_idx) > 0:
        print(
            'warning: loss exceeds threshold of {:.2f} in total {} time(s); values will be clipped'.format(
                loss_clipping,
                len(clip_idx)))

    y[clip_idx] = loss_clipping

    options = {}
    options['title'] = title
    options['xlabel'] = r'iteration'
    options['ylabel'] = r'loss'

    fig, ax = plt.subplots(1, 1)
    ax.semilogx(x, y, 'b')
    ax.set_xlabel(options['xlabel'])
    ax.set_ylabel(options['ylabel'])

    return fig, ax


def dist(dist, title=''):
    """Given dist, plot histogram"""
    options = {}
    options['title'] = title
    options['xlabel'] = r'bin'
    options['ylabel'] = r'distance'

    fig = plt.figure()
    ax = fig.add_subplot(111)
    n_samples = len(dist)
    ax.hist(dist, bins=int(np.sqrt(n_samples)))
    ax.set_xlabel(options['xlabel'])
    ax.set_ylabel(options['ylabel'])
    ax.set_title(options['title'])
    return fig, ax


def info(info, html=False, title=None):
    """Given info dict, produce info text"""
    if title is None:
        infotext = u''
    else:
        if html:
            infotext = u'<b>{}</b><br>'.format(title)
        else:
            infotext = u'{}\n'.format(title)

    for key, value in info.items():
        if key not in ['losses']:
            infotext += u'{} : {}'.format(key, value)
            if html:
                infotext += '<br>'
            else:
                infotext += '\n'

    return infotext


def probs2contours(probs, levels):
    """Takes an array of probabilities and produces an array of contours at specified percentile levels

    Parameters
    ----------
    probs : array
        Probability array. doesn't have to sum to 1, but it is assumed it contains all the mass
    levels : list
        Percentile levels, have to be in [0.0, 1.0]

    Return
    ------
    Array of same shape as probs with percentile labels
    """
    # make sure all contour levels are in [0.0, 1.0]
    levels = np.asarray(levels)
    assert np.all(levels <= 1.0) and np.all(levels >= 0.0)

    # flatten probability array
    shape = probs.shape
    probs = probs.flatten()

    # sort probabilities in descending order
    idx_sort = probs.argsort()[::-1]
    idx_unsort = idx_sort.argsort()
    probs = probs[idx_sort]

    # cumulative probabilities
    cum_probs = probs.cumsum()
    cum_probs /= cum_probs[-1]

    # create contours at levels
    contours = np.ones_like(cum_probs)
    levels = np.sort(levels)[::-1]
    for level in levels:
        contours[cum_probs <= level] = level

    # make sure contours have the order and the shape of the original
    # probability array
    contours = np.reshape(contours[idx_unsort], shape)

    return contours

def plot_diag_axis(ax, i, pdfs, colors, samples, lims, gt, bins, resolution, hist_color, ticks, labels_params, fontscale):
    if samples is not None:
        ax.hist(samples[...,i], bins=bins, normed=True, color=hist_color, edgecolor=hist_color) 

    xx = np.linspace(lims[i,0], lims[i,1], resolution)

    for pdf, col in zip(pdfs, colors):
        if pdf is not None:
            pp = pdf.eval(xx, ii=[i], log=False)
            ax.plot(xx, pp, color=col)

    ax.set_xlim(lims[i])
    ax.set_ylim([0, ax.get_ylim()[1]])

    if gt is not None:
        ax.axvline(gt[i], color='r')


    if labels_params is not None:
        ax.set_xlabel(labels_params[i], fontsize=fontscale * 20)
    else:
        ax.set_xlabel("")

    if ticks:

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)

        ax.set_yticks([])        
        ax.set_xticks(np.round(lims[i]*10)/10)

        if i < 2:
            ax.set_xticklabels([r"$%d^\circ$" %np.int(np.round(lim/np.pi * 180)) for lim in lims[i]])
        #ax.get_yaxis().set_tick_params(
        #    which='both', direction='out', labelsize=fontscale * 15)
        #ax.get_xaxis().set_tick_params(
        #    which='both', direction='out', labelsize=fontscale * 15)
#                         axes[i,j].locator_params(nbins=3)
        #axes[i,j].set_xticks(np.linspace(
        #    lims[i, 0]+0.15*np.abs(lims[i, 0]-lims[j, 1]), lims[j, 1]-0.15*np.abs(lims[i, 0]-lims[j, 1]), 2))
        #axes[i,j].set_yticks(np.linspace(0+0.15*np.abs(0-max(pp)), max(pp)-0.15*np.abs(0-max(pp)), 2))
        #axes[i,j].xaxis.set_major_formatter(
            #mpl.ticker.FormatStrFormatter('%.1f'))
        #axes[i,j].yaxis.set_major_formatter(
            #mpl.ticker.FormatStrFormatter('%.1f'))
    else:
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)        


    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    ax.set_aspect((x1 - x0) / (y1 - y0))


def plot_marg_axes(ax, i, j, pdf, samples, lims, gt, bins, resolution, cmap, contours, contour_levels, contour_colors, scatter, scatter_color, scatter_alpha): 
    assert i != j

    if samples is not None and not scatter:
        H, xedges, yedges = np.histogram2d(samples[...,i], samples[...,j], bins=bins, range=[lims[i], lims[j]], normed=True)
        ax.imshow(H, origin='lower', extent=[yedges[0], yedges[-1], xedges[0], xedges[-1]], cmap=cmap)

    if pdf is not None:
        xx = np.linspace(lims[i, 0], lims[i, 1], resolution)
        yy = np.linspace(lims[j, 0], lims[j, 1], resolution)
        X, Y = np.meshgrid(xx, yy)
        xy = np.concatenate([X.reshape([-1, 1]), Y.reshape([-1, 1])], axis=1)
        pp = pdf.eval(xy, ii=[i, j], log=False).reshape(X.shape)

        if contours:
            ax.contour(Y, X, probs2contours(pp, contour_levels), contour_levels, colors=contour_colors)
        else:
            ax.imshow(pp.T, origin='lower', cmap=cmap, extent=[lims[j, 0], lims[j, 1], lims[i, 0], lims[i, 1]], aspect='auto', interpolation='none')

    if samples is not None and scatter:
        ax.plot(samples[...,j], samples[...,i], '.', c=scatter_color, alpha=scatter_alpha, ms=2)
        
    ax.set_xlim(lims[j])
    ax.set_ylim(lims[i])

    if gt is not None:
        ax.plot(gt[j], gt[i], 'r.', ms=8, markeredgewidth=0.0)

    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    #ax.set_axis_off()

    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    ax.set_aspect((x1 - x0) / (y1 - y0))

def plot_pdf(pdf1=None, pdf2=None, samples=None, lims=None, gt=None, 
             contours=False, contour_levels=(0.68, 0.95), contour_colors=('w','y'),
             resolution=500, labels_params=None, ticks=False, diag_only=False,
             figsize=(5, 5), fontscale=1, marginals=None, scatter=False, scatter_color='gray', scatter_alpha=0.2,
             bins=100, cmap=None,
             hist_color='gray', pdf1_color='b', pdf2_color='g'):
    """Plots marginals of a pdf, for each variable and pair of variables.

    Parameters
    ----------
    pdf1 : object
    lims : array
    pdf2 : object (or None)
        If not none, visualizes pairwise marginals for second pdf on lower diagonal
    marginals : array (or None)
        Array of indices 0 .. ndim-1 with coordinates to plot
    contours : bool
    levels : tuple
        For contours
    resolution
    labels_params : array of strings
    ticks: boo, figsize=figsize)
        If True, includes ticks in plots
    diag_only : bool
    diag_only_cols : int
        Number of grid columns if only the diagonal is plotted
    diag_only_rows : int
        Number of grid rows if only the diagonal is plotted
    fontscale: int
    samples: array
        If given, samples of a distribution are plotted along `pdf`.
        If given, `pdf` is plotted with default `levels` (0.68, 0.95), if provided `levels` is None.
        If given, `lims` is overwritten and taken to be the respective
        limits of the samples in each dimension.
    hist_color
        Color for histogram (if samples are given)
    pdf1_color : str
        color 2
    pdf2_color : str
        color 3 (for pdf2 if provided)
    """

    pdfs = (pdf1, pdf2)
    colrs = (pdf1_color, pdf2_color)
    
    ndim = None

    if pdf1 is not None:
        ndim = pdf1.ndim

    if pdf2 is not None:
        if ndim is not None:
            assert pdf2.ndim == ndim
        else:
            ndim = pdf2.ndim

    if samples is not None:
        samples = np.asarray(samples)

        if ndim is not None:
            assert samples.shape[-1] == ndim
        else:
            ndim = samples.shape[-1]

    if ndim == None:
        raise ValueError("No pdf and no samples given to plot")

    if samples is not None:
        sample_shape = np.asarray(samples).shape
        assert len(sample_shape) == 2 and sample_shape[-1] == ndim

        if not scatter:
            contours = True
        
        if lims is None:
            lims_min = np.min(samples, axis=0)
            lims_max = np.max(samples, axis=0)
    
            lims = np.asarray([lims_min, lims_max]).T

    assert lims is not None, "No limits specified and no samples given"
    lims = np.asarray(lims)
    if lims.ndim == 1:
        lims = np.tile(lims, [ndim, 1])

    marginal_mask = np.ones(ndim, bool)
    if marginals is not None:
        marginals = np.asarray(marginals)
        assert np.all(0 <= marginals) and np.all(marginals < ndim)
        for m in marginals:
            marginal_mask[m] = False

    n_marg = np.count_nonzero(marginal_mask)

    n_rows = n_marg
    n_cols = n_marg

    fig, axes = plt.subplots(n_rows, n_cols, facecolor='white', figsize=figsize)
    axes = axes.reshape(n_rows, n_cols)

    for i in range(n_rows):
        for j in range(n_cols):
            if i == j:
                plot_diag_axis(ax=axes[i,j], 
                               i=i, 
                               pdfs=pdfs, 
                               colors=colrs, 
                               samples=samples, 
                               lims=lims, 
                               gt=gt, 
                               bins=bins, 
                               resolution=resolution,
                               ticks=ticks,
                               hist_color=hist_color, 
                               labels_params=labels_params, 
                               fontscale=fontscale)

            else:
                if i < j:
                    pdf = pdfs[0]
                else:
                    pdf = pdfs[1]

                if diag_only or pdf is None and i > j:
                    axes[i, j].get_yaxis().set_visible(False)
                    axes[i, j].get_xaxis().set_visible(False)
                    axes[i, j].set_axis_off()
                    continue

                plot_marg_axes(ax=axes[i,j], 
                               i=i, j=j,
                               pdf=pdf, 
                               samples=samples, 
                               lims=lims, 
                               gt=gt, 
                               bins=bins, 
                               resolution=resolution,
                               cmap=cmap,
                               contours=contours,
                               contour_levels=contour_levels,
                               contour_colors=contour_colors,
                               scatter=scatter,
                               scatter_color=scatter_color,
                               scatter_alpha=scatter_alpha)

    # plt.tight_layout()
    return fig, axes

def plot_hist_marginals(data, lims=None, gt=None):
    """Plots marginal histograms and pairwise scatter plots of a dataset"""
    n_bins = int(np.sqrt(data.shape[0]))

    if data.ndim == 1:
        fig, ax = plt.subplots(1, 1, facecolor='white')
        ax.hist(data, n_bins, normed=True)
        ax.set_ylim([0, ax.get_ylim()[1]])
        if lims is not None:
            ax.set_xlim(lims)
        if gt is not None:
            ax.vlines(gt, 0, ax.get_ylim()[1], color='r')

    else:
        n_dim = data.shape[1]
        fig, ax = plt.subplots(n_dim, n_dim, facecolor='white')
        ax = np.array([[ax]]) if n_dim == 1 else ax

        if lims is not None:
            lims = np.asarray(lims)
            lims = np.tile(lims, [n_dim, 1]) if lims.ndim == 1 else lims

        for i in range(n_dim):
            for j in range(n_dim):

                if i == j:
                    ax[i, j].hist(data[:, i], n_bins, normed=True)
                    ax[i, j].set_ylim([0, ax[i, j].get_ylim()[1]])
                    if lims is not None:
                        ax[i, j].set_xlim(lims[i])
                    if gt is not None:
                        ax[i, j].vlines(
                            gt[i], 0, ax[i, j].get_ylim()[1], color='r')

                else:
                    ax[i, j].plot(data[:, i], data[:, j], 'k.', ms=2)
                    if lims is not None:
                        ax[i, j].set_xlim(lims[i])
                        ax[i, j].set_ylim(lims[j])
                    if gt is not None:
                        ax[i, j].plot(gt[i], gt[j], 'r.', ms=8)

    return fig, ax
