import numpy as np
import matplotlib.pylab as plt
import pandas as pd
import seaborn as sns

from pystackreg import StackReg
from scipy.stats import zscore
from scipy.optimize import linear_sum_assignment


from itertools import combinations, product
import cupy as cp
from scipy.stats import spearmanr

import networkx as nx

from pathlib import Path
from datetime import datetime
from joblib import Parallel, delayed, parallel_backend
import tempfile
import imageio

################
## data handling

def load_data(paths, only_cells=True):

    refs, rois, meds = [], [], []
    for p in paths:
    
        # load npy files
        stat = np.load(p / 'stat.npy', allow_pickle=True)
        iscell = np.load(p / 'iscell.npy', allow_pickle=True)
        ops = np.load(p / 'ops.npy', allow_pickle=True)

        # load ref image, set resolution
        img = ops.item()['meanImgE']
        res = img.shape
        refs.append(img)

        # load rois/meds
        rs, ms = [], []
        for i, j in zip(stat, iscell):
            if not only_cells or j[0]:
                r = np.zeros(shape=res)
                for x, y, l in zip(i['xpix'], i['ypix'], i['lam']):
                    r[y, x] = l
                rs.append(r)
                y, x = i['med']
                ms.append((x, y))

        rois.append(rs)
        meds.append(ms)

    return refs, rois, meds

def get_roi_info(ps):

    df = pd.DataFrame() 

    for n, p in enumerate(ps):
        
        # which rois are cells
        iscell = np.load(p / 'iscell.npy', allow_pickle=True)
        cell_mask = iscell[:, 0].astype(bool)

        # rate
        spks = np.load(p / 'spks.npy', allow_pickle=True)
        r = spks.mean(axis=1)[cell_mask]

        # red channel probaility
        redcell = np.load(p / 'redcell.npy', allow_pickle=True)
        red = redcell[:, 1][cell_mask]

        # load meds
        stat = np.load(p / 'stat.npy', allow_pickle=True)
        med = [ i['med'] for i in stat[cell_mask] ]
        y = [ i[0] for i in med ]
        x = [ i[1] for i in med ]


        # collect in dataframe
        d = pd.DataFrame(data={
            's'         : n, 
            'r'         : np.arange(cell_mask.sum()),
            'rate'      : r,
            'redcell'   : red,
            'x'         : x,
            'y'         : y,
        })

        df = pd.concat([df, d], ignore_index=True)
    
    return df


def get_nsess_dict(df):

    cols = df.loc[:, :'n'].columns
    # last column is number of sesseions
    col_n = cols[-1]
    # other columns are session indices
    col_sess = cols[:-1]

    # dict of dicts: session -> roi -> number of connected sess
    d = { int(s): dict() for s in col_sess }

    # cycle through each connected component
    for i in df.index:

        # number of connected sessions
        n = df.loc[i, col_n].item()

        # cycle through each line
        for s, r in df.loc[i, col_sess].to_dict().items():
            # skip nans
            if r == r:
                d[int(s)][int(r)] = n
    return d

def add_nsess(df_roi, df_matched):

    # mapping from roi to number of connected sessions
    d_nsess = get_nsess_dict(df_matched)
    # return 1 if not found
    map_d_nses = lambda x: d_nsess[x[0]].get(x[1], 1)

    # apply mapping
    ds = df_roi.loc[:, ['s', 'r']].apply(map_d_nses, axis=1)
    df_roi.loc[:, 'n'] = ds



def get_rate_red(df_matched, df_roi):

    # get dfs with same shape as df_matched but data is ROI rate/ red channel prob
    # empty dataframes
    df_rate = pd.DataFrame(index=df_matched.index, columns=df_matched.columns)
    df_red = df_rate.copy()

    # for easy access of rois rate/red ch
    gr = df_roi.groupby(['s', 'r'])

    # make sure last column is 'n'
    df_matched = df_matched.loc[:, :'n']
    # ignore last column
    cols = df_matched.columns[:-1]
    
    # cycle through all matches
    for i in df_matched.index:
        
        # cycle through roi for each match
        row = df_matched.loc[i, cols]

        for k, v in row.dropna().to_dict().items():
            s, r = int(k), int(v)

            # corresning entrly in df_roi
            d = gr.get_group((s, r))

            rate = d.loc[:, 'rate'].item()
            df_rate.loc[i, str(s)] = rate

            red = d.loc[:, 'redcell'].item()
            df_red.loc[i, str(s)] = red

            n = d.loc[:, 'n'].item()
            df_rate.loc[i, 'n'] = n
            df_red.loc[i, 'n'] = n

    # add mean and SD column
    for df in [df_rate, df_red]:
        cols = df.columns[:-1]
        df.loc[:, 'mean'] = df.loc[:, cols].mean(axis=1)
        df.loc[:, 'SD'] = df.loc[:, cols].std(axis=1)

    return df_rate, df_red

def add_skipped(df):

    cols = df.loc[:, :'n'].columns[:-1]

    for idx in df.index:

        ds = df.loc[idx, cols]
        x = (ds == ds).astype(int)

        # number of missing sessions
        i_1 = np.flatnonzero(x == 1)
        i_i, i_f = i_1[0], i_1[-1]

        i_0 = np.flatnonzero(x == 0)
        n_0 = np.sum((i_0 > i_i) & (i_0 < i_f))
        
        df.loc[idx, 'n_skip'] = n_0

        # count interruptions
        d = np.diff(x, append=0)
        off = (d == -1).sum()
        df.loc[idx, 'n_dis'] = off - 1

def get_skipped_sessions(df):

    cols = df.loc[:, :'n'].columns

    s_off, n_off = [], []
    for idx in df.index:

        n = df.loc[idx, cols[-1]]
        ds = df.loc[idx, cols[:-1]]
        x = (ds == ds).astype(int)

        # sessions that interrupt
        d = np.diff(x)
        i = np.flatnonzero(d == -1) + 1
        s_off.extend(i)
        n_off.extend([n]*len(i))

    df = pd.DataFrame(data={'session': s_off, 'n': n_off})
    
    return df

def count_nonunique(df):

    cols = df.loc[:, :'n'].columns
    df = df.loc[:, cols]

    dss = []
    for i in df.loc[:, 'n'].unique():
        df_ = df.loc[ df.loc[:, 'n'] >= i ].drop(columns='n')
        ds = (df_ == df_).sum()
        ds.name = i
        dss.append(ds)

    df = pd.DataFrame(dss)

    return df

###############
## registration

def get_tmats(imgs, reg, n0=0, do_zscore=True, n_proc=-1):

    if do_zscore:
        imgs = [zscore(img, axis=None) for img in imgs]

    img0 = imgs[n0]
    sr = StackReg(reg)

    with parallel_backend('loky', n_jobs=n_proc):
    
        tmats = Parallel()(
            delayed(sr.register)(img0, img) for img in imgs
        )
    # tmats = [ np.round(t, 0) for t in tmats ]

    return tmats


def align_refs(refs, tmats, reg, n_proc=-1):
    
    sr = StackReg(reg)
    with parallel_backend('loky', n_jobs=n_proc):

        refs_al = Parallel()(
            delayed(sr.transform)(i, j) for i, j in zip(refs, tmats)
        )

    return refs_al



def align_rois(rois, tmats, reg, n_proc=-1):

    sr = StackReg(reg)
    rois_al = []
    for rs, tmat in zip(rois, tmats):
        with parallel_backend('loky', n_jobs=n_proc):
            rs_al = Parallel()(
                delayed(sr.transform)(r, tmat) for r in rs
            )
        rois_al.append(rs_al)

    return rois_al

##############
## correlation

def calculate_corrcoef(l1, l2, corr, use_gpu=True, n_proc=-1):

    # initiate corrcoeff matrix
    n1, n2 = len(l1), len(l2)
    cc = np.empty((n1, n2))

    # flatten 2d images
    l1 = [ i.flatten() for i in l1 ]
    l2 = [ i.flatten() for i in l2 ]

    # all unique index pairs between l1 and l2
    i12 = [ *product(range(n1), range(n2)) ]

    if use_gpu:
    
        # convert to cupy array
        l1 = [ cp.array(i) for i in l1 ]
        l2 = [ cp.array(i) for i in l2 ]

        # cycle through all pairs
        for i1, i2 in i12:
            # calculate corrcoef
            if corr == 'pearson':
                c = cp.corrcoef(l1[i1], l2[i2])[0, 1]
                cc[i1, i2] = c
            else:
                raise NotImplementedError

    else:

        # run on n_proc CPU cores
        with parallel_backend('loky', n_jobs=n_proc):
            
            if corr == 'pearson':
                fun = lambda x, y: np.corrcoef(x, y)[0, 1]
            elif corr == 'spearman':
                fun = lambda x, y: spearmanr(x, y).statistic

            res = Parallel()(
                delayed(fun)(l1[i1], l2[i2]) for i1, i2 in i12
                )

        # write list to array
        for j, (i1, i2) in enumerate(i12):
            cc[i1, i2] = res[j]

    return cc

def corrcoeff_wrapper(rois, ps, path, corr='pearson', use_gpu=False, n_proc=-1):

    # create output folder
    path.mkdir(exist_ok=True, parents=True)

    # convenience function to define output file name
    p2str = lambda p: '-'.join(p.parts[-4:-2])

    n_sess = len(rois)
    for i, j in combinations(range(n_sess), r=2):
        
        # define output file name
        si, sj = p2str(ps[i]), p2str(ps[j])
        npy = Path(path) / '{}_{}.npy'.format(si, sj)
        if npy.is_file():
            print('INFO: file {} exists, skipping calculation'.format(npy))
            continue

        # select two sessions
        ri, rj = rois[i], rois[j]

        # get corrcoef matrix
        cc = calculate_corrcoef(ri, rj, corr=corr, use_gpu=use_gpu, n_proc=n_proc)

        # save to disk
        print('INFO: {}'.format(datetime.now()))
        print('INFO: saving {}'.format(npy))
        np.save(npy, cc)


def load_corrcoef(ps, path):
    
    p2str = lambda p: '-'.join(p.parts[-4:-2])
    
    d = dict()

    for i1, i2 in combinations(range(len(ps)), r=2):

        si, sj = p2str(ps[i1]), p2str(ps[i2])
        npy = Path(path) / '{}_{}.npy'.format(si, sj)
        if npy.is_file():
            d[(i1, i2)] = np.load(npy)

    return d

def assign_matches(d_cc):

    df = pd.DataFrame()

    for (sa, sb), cc in d_cc.items():
        
        # linear sum assignment solves matching problem
        ia, ib = linear_sum_assignment(cc, maximize=True)
        cc_m = cc[ia, ib]

        # sort by increasing cc
        i = np.argsort(cc_m)
        ia, ib, cc_m = ia[i], ib[i], cc_m[i]

        d = pd.DataFrame(data={
            'sa' : sa,
            'sb' : sb,
            'ra' : ia,
            'rb' : ib,
            'cc' : cc_m,
        })

        df = pd.concat([df, d], ignore_index=True)
    
    return df

def compare_df_rois(df, rois):

    # rois in dataframe
    gr_a = df.groupby(['sa', 'ra'])
    gr_b = df.groupby(['sb', 'rb'])
    s_df = gr_a.groups.keys() | gr_b.groups.keys()
    n_df = len(s_df)
    
    # rois in list
    n_rois = sum([len(r) for r in rois])

    print('ROIs in df   : {}'.format(n_df))
    print('ROIs in rois : {}'.format(n_rois))
    print('>> {} ROIs lost in linear assignment problem'.format(n_rois - n_df))


########
## Graph 

def construct_graph(df):

    # convert dataframe to list of weighted edges
    edges = []
    for k, df  in df.groupby(['sa', 'ra', 'sb', 'rb']):
        sa, ra, sb, rb = k
        cc = df.loc[:, 'cc'].item()
        e = ((sa, ra), (sb, rb), cc)
        edges.append(e)

    # construct graph from edges
    G = nx.Graph()
    G.add_weighted_edges_from(edges)

    # add notde attributes
    for n in G.nodes:
        s, r = n
        G.nodes[n]['s'] = s
        G.nodes[n]['r'] = r

    return G


def is_complete(G):

    # number of nodes
    n = len(G)
    m = (n * (n - 1)) // 2

    # number of edges
    e = G.number_of_edges()
    
    return e == m

flatten2set = lambda x: { i for j in x for i in j }

def filter_edges(G, thresh):
    
    # select edges below thresh
    weak_edges = [ (u, v) for u, v, w in G.edges(data='weight') if w < thresh ]

    # remove from graph
    G.remove_edges_from(weak_edges)

def remove_weakest_edge(G):

    # get edge with lowest weight
    d = nx.get_edge_attributes(G, 'weight')
    k = min(d, key=d.get)
    
    # remove from graph
    G.remove_edge(*k)

    v = d[k]

    return v


def pop_complete(G, n=None):

    # collect complete
    complete = []

    # cycle through connected
    for cc in nx.connected_components(G):

        g = G.subgraph(cc)

        if n:
            # only pop CCs with a given n
            if g.number_of_nodes() != n:
                continue

        # check if complete
        if is_complete(g):
            complete.append(cc)

    # remove complete CCs from graph
    G.remove_nodes_from(flatten2set(complete))

    return complete


def collect_connected(G):

    # number of sessions
    n_sess = len({ i for _, i in G.nodes.data('s') })

    # list to collect complete connected components
    complete = []

    # step 1
    # collect only complete with n = n_sess
    for t in np.linspace(1, .5, 101):

        G_sub = G.copy()

        G_sub.remove_nodes_from(flatten2set(complete))

        filter_edges(G_sub, thresh=t)

        l = pop_complete(G_sub, n=n_sess)
        complete.extend(l)

    print('INFO: found {} complete connected components with n = {}'.format(len(complete), n_sess))

    # step 2
    # collect all complete CC at thresh = 0.5
    thresh = 0.5
    filter_edges(G_sub, thresh=thresh)

    l = pop_complete(G_sub)
    complete.extend(l)

    print('INFO: found {} futher complete connected components with weights > {}'.format(len(l), thresh))

    # step 3
    # remove one edge at a time to collect complete CC
    n = 0
    while G_sub.nodes:

        w = remove_weakest_edge(G_sub)

        l = pop_complete(G_sub)
        complete.extend(l)

        n += len(l)

    print('INFO: found {} futher complete connected components after removing one edge at a time (weight of last edge removed: {})'.format(n, w))

    # construct dataframe
    df = pd.DataFrame(index=range(len(complete)), columns=range(n_sess))

    for i, cc in enumerate(complete):

        df.loc[i, 'n' ] = len(cc)
        for c in cc:
            df.loc[i, c[0] ] = c[1]

    # add size column
    df.loc[:, 'n'] = df.loc[:, 'n'].astype(int)
    # sort
    df = df.sort_values(by='n', ascending=False)
    # convert to strings for easy parquetting
    df.columns = df.columns.astype(str)

    return df


###########
## plotting
def write_gif(imgs, gif, fps=2):
    '''Generate GIF from list of files

    Parameters
    ----------
    imgs : list
        Files to be concatenated in GIF
    gif : path-like
        Name of the output file, has to have .gif suffix
    fps : int, optional
        Frames per second in GIF, by default 1
    '''

    with imageio.get_writer(gif, mode='I', duration=1000 / fps, loop=0) as writer:
        for img in imgs:
            img = imageio.imread(img)
            writer.append_data(img)

def write_fig(img, title, tmpdir, cutoff=5):
              
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_title(title)

    lp, hp = np.nanpercentile(img, [cutoff, 100 - cutoff])
    ax.imshow(img, vmin=lp, vmax=hp)

    fig.tight_layout()
    png = tempfile.NamedTemporaryFile(suffix='.png', dir=tmpdir, delete=False).name
    fig.savefig(png)
    plt.close(fig)

    return png


def plot_gif(imgs, gif, n_proc=-1):

    with tempfile.TemporaryDirectory() as tmpdir:

        with parallel_backend('loky', n_jobs=n_proc):
            pngs = Parallel()(
                delayed(write_fig)(img, 'img {:03d}'.format(i), tmpdir) for (i, img) in enumerate(imgs)
            )

        write_gif(pngs, gif)

def plot_rois(df, rois, ab, tmpdir, pxl=25):

    # initiate figure
    ncols = 10
    nrows = int(np.ceil(len(df) / ncols))
    fig, axmat = plt.subplots(nrows=nrows, ncols=ncols, figsize=(8, nrows*1.5))
    for ax in axmat.flatten():
        ax.axis('off')

    # cycle through rows of df
    for idx, ax in zip(df.index, axmat.flatten()):

        # get cc
        cc = df.loc[idx, 'cc']

        # center based on max in session a
        s, r = df.loc[idx, [ 'sa', 'ra']].astype(int)
        roi = rois[s][r]
        cnt = [ i[0] for i in np.where(roi == roi.max())]

        # roi in current session (a or b)
        s, r = df.loc[idx, [ 's' + ab, 'r' + ab]].astype(int)
        roi = rois[s][r]

        # plot
        ax.imshow(roi)

        ax.set_xlim((cnt[1] - pxl, cnt[1] + pxl))
        ax.set_ylim((cnt[0] - pxl, cnt[0] + pxl))

        ax.set_title('{:1.2f}'.format(cc))

    fig.tight_layout()
    png = tempfile.NamedTemporaryFile(suffix='.png', dir=tmpdir, delete=False).name
    fig.savefig(png)
    plt.close(fig)

    return png


def plot_roi_pairs(df, rois, gif):

    # work within tmpdir to avoid having to delete files
    with tempfile.TemporaryDirectory() as tmpdir:

        # use 2 cpus on indi
        with parallel_backend('loky', n_jobs=2):
            pngs = Parallel()(
                delayed(plot_rois)(df, rois, ab, tmpdir) for ab in 'ab'
            )

        write_gif(pngs, gif)

def plot_session_pairs(df, normalize=None):

    # only session columns
    cols = df.loc[:, :'n'].columns[:-1]

    # upper triangle
    x = np.empty((len(cols), len(cols)))
    x[:] = np.nan
    for ca, cb in combinations(cols, r=2):
        dsa = df.loc[:, ca]
        dsb = df.loc[:, cb]
        a = dsa == dsa
        b = dsb == dsb
        m = (a * b).sum()
        x[int(ca), int(cb)] = m

    # diagonal
    y = np.empty((len(cols), len(cols)))
    y[:] = np.nan
    for c in cols:
        ds = df.loc[:, c]
        m = (ds == ds).sum()
        y[int(c), int(c)] = m
    
    if normalize == 'row':
        y_diag = np.diagonal(y)
        Y = np.array([ y_diag ] * len(y_diag))
        x = x / Y
        y = y / Y
        fmt = '.2f'

    elif normalize == 'mean':
        y_diag = np.diagonal(y)
        Y_diag = np.array([ y_diag ] * len(y_diag))
        Y = ( Y_diag + Y_diag.T ) / 2
        x = x / Y
        y = y / Y
        fmt = '.2f'

    else:
        fmt='.0f'

    # plot
    fig, ax = plt.subplots(figsize=(len(cols)/1.5, len(cols)/2))

    sns.heatmap(x, ax=ax, annot=True, fmt=fmt, linewidths=0.01, square=True, cbar_kws={'label': 'ROI pairs'})
    sns.heatmap(y, ax=ax, annot=True, fmt=fmt, linewidths=0.01, square=True, cmap='binary', vmin=0, cbar=False)

    ax.set_xlabel('session a')
    ax.set_ylabel('session b')

    fig.tight_layout()
