# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: everything
#     language: python
#     name: python3
# ---

# %%
# %load_ext autoreload
# %autoreload 2

import numpy as np
import matplotlib.pylab as plt
import pandas as pd
import seaborn as sns
import src.utils as utl

from pystackreg import StackReg
import networkx as nx

from pathlib import Path

# %% [markdown]
# # Prepare data

# %% [markdown]
# ## load suite2p data

# %%
# define folders
ps = [ p.parent for p in Path('./data/ANMP214/').glob('**/??/**/stat.npy') ]

p_out = Path('./output/')
p_out.mkdir(exist_ok=True, parents=True)

# load refs, rois, meds
refs, rois, meds = utl.load_data(ps, only_cells=False)

# load ROI info
df_roi = utl.get_roi_info(ps)

# save
df_roi.to_parquet(p_out / 'df_roi_all_info.parquet')

# %% [markdown]
# ## align ref images and rois

# %%
# choose method for registration: https://pystackreg.readthedocs.io/en/latest/readme.html#usage
reg = StackReg.BILINEAR
# get tranformation matrices
tmats = utl.get_tmats(refs, reg)

# align ref images
refs = utl.align_refs(refs, tmats, reg)
# align rois
rois = utl.align_rois(rois, tmats, reg)

# plot aligned refs to confirm (always inspect registration!!!)
utl.plot_gif(refs, p_out / 'aligned_refs.gif')

# %% [markdown]
# ## Calculate CCFs

# %%
# calculate corrcoef, store on disk
utl.corrcoeff_wrapper(rois, ps, corr='pearson', use_gpu=True, path=p_out / 'pearson_all_rois')

# %%
# load corrcoef from disk
d_cc = utl.load_corrcoef(ps,  p_out / 'pearson_all_rois')

# %% [markdown]
# # Match ROI pairs

# %%
# assign roi matches based on CC
df_all = utl.assign_matches(d_cc)

# %%
# distribution of CCs between matched ROIs
sns.histplot(data=df_all, x='cc')

# %%
# plot borderline cases
df = df_all
# select rois with intermediate CC
df = df.loc[ (df.loc[:, 'cc'] > 0.3) & (df.loc[:, 'cc'] < 0.7) ]
# select smaller number
df = df.sort_values('cc').loc[::100, :]
# create gif
utl.plot_roi_pairs(df, rois, p_out / 'roi_pairs.gif')

# %%
# check how many ROIs had exactly 0 overlap
utl.compare_df_rois(df_all, rois)

# %%
# save
df_all.to_parquet(p_out / 'pearson_all_matched.parquet')

# %%
# load
df_all = pd.read_parquet(p_out / 'pearson_all_matched.parquet')

# %% [markdown]
# ## find global matches

# %%
# combine roi pairs in singe graph
G = utl.construct_graph(df_all)

# %%
# iteratively filter weights to get complete connected components
df_matched = utl.collect_connected(G)

# %%
# distribution of matches appearing in n sessions
sns.histplot(data=df_matched, x='n', discrete=True)

# %%
# distribution of matches appearing in n sessions
sns.histplot(data=df_matched, x='n', discrete=True)

# %%
# which session pairs share how many ROIs
utl.plot_session_pairs(df_matched)

# %%
# which session pairs share how many ROIs
utl.plot_session_pairs(df_matched)

# %%
# save
df_matched.to_parquet(p_out / 'complete.parquet')

# %% [markdown]
# # Analyze matches

# %%
# load matched ROIS
df_matched = pd.read_parquet('./data/ANMP214/corrcoef/complete.parquet')
utl.add_skipped(df_matched)

# load ROI info
df_roi = pd.read_parquet('./data/df_roi_info.parquet')
utl.add_nsess(df_roi, df_matched)

# dataframes with rate/red channel prob with same structure as df_matched
df_rate, df_red = utl.get_rate_red(df_matched, df_roi)

# %% [markdown]
# ## How many sessions? 

# %%
# relationship between rate and red channel probability
fig, ax = plt.subplots(figsize=(10, 5))
sns.scatterplot(data=df_roi, x='rate', y='redcell', hue='n')
ax.set_xlabel('average firing rate')
ax.set_ylabel('red channel probability')

# %%
# distribution of rates split by number of sessions
fig, axarr = plt.subplots(ncols=2, figsize=(20, 5))
ax = axarr[0]
sns.histplot(data=df_roi, ax=ax, x='rate', y='n', discrete=(False, True))
ax.set_title('average firing rate')

ax = axarr[1]
sns.histplot(data=df_roi, ax=ax, x='redcell', y='n', discrete=(False, True))
ax.axvline(0.35, c='gray', ls='--')
ax.axvline(0.65, c='gray', ls='--')
ax.set_title('red channel probability')


# %%
# standard deviation
fig, axarr = plt.subplots(ncols=2, figsize=(20, 5))
ax = axarr[0]
sns.histplot(data=df_rate, ax=ax, x='SD', y='n', discrete=(False, True))
ax.set_title('SD of rate')

ax = axarr[1]
sns.histplot(data=df_red, ax=ax, x='SD', y='n', discrete=(False, True))
ax.set_title('SD of red channel probability')


# %% [markdown]
# ## time-domain sparsity

# %%
# skipped sessions
df = df_matched
df = df.loc[ df.loc[:, 'n'] > 1 ]

fig, ax = plt.subplots(figsize=(10, 5))
sns.histplot(df, ax=ax, x='n_skip', y='n', discrete=True)
ax.set_xlabel('number of skipped sessions')

# %%
# interruptions
df = df_matched
df = df.loc[ df.loc[:, 'n'] > 1 ]

fig, ax = plt.subplots(figsize=(10, 5))
sns.histplot(df, ax=ax, x='n_dis', y='n', discrete=True)
ax.plot(np.arange(0, 14), c='gray', ls='--')
ax.plot(np.arange(26, 12, -1), c='gray', ls='--')
ax.set_xlabel('number of interruptions')

# %% [markdown]
# ## Are there "bad" sessions?

# %%
# skipped sessions
df = utl.get_skipped_sessions(df_matched)
fig, axarr = plt.subplots(ncols=2, figsize=(20, 5))

ax = axarr[0]
sns.histplot(df, ax=ax, x='session', discrete=True)

ax = axarr[1]
sns.histplot(df, ax=ax, x='session', y='n', discrete=True)

# %% [markdown]
# # consecutive sessions

# %%
df = utl.count_nonunique(df_matched)
fig, ax = plt.subplots(figsize=(20, 15))

sns.heatmap(data=df, ax=ax, annot=True, fmt='.0f', square=True, cbar_kws={'label': 'number of non-unique ROIs'})

# %%
# which session pairs share how many ROIs
utl.plot_session_pairs(df_matched, normalize='row')


# %% [markdown]
# ## TODO: anatomical distance

# %%
def add_nn(df, min_n=1):

    df_ = df.loc[ df.loc[:, 'n'] > min_n ]

    for _, df_s in df_.groupby('s'):
        XY = df_s.loc[:, ['x', 'y']]

        for _, df_r in df_s.groupby('r'):
            xy = df_r.loc[:, ['x', 'y']].values
            # i = np.argsort(np.linalg.norm(XY - xy, axis=1))[1]
            d = np.sort(np.linalg.norm(XY - xy, axis=1))[1]
            df.loc[df_r.index, 'dNN{}'.format(min_n)] = d


add_nn(df_roi, min_n=0)
add_nn(df_roi, min_n=1)
add_nn(df_roi, min_n=2)


# %%
fig, ax = plt.subplots(figsize=(10, 5))

sns.histplot(df_roi, x='dNN0', y='s', discrete=(False, True))

# %% [markdown]
# # show ROIs

# %%
import io
import imageio
ps = [ p.parent for p in Path('./data/ANMP214/A214-20221214/02/').glob('**/stat.npy') ]

# load refs, rois, meds
refs, rois, meds = utl.load_data(ps, only_cells=False)

# load ROI info
df_roi = utl.get_roi_info(ps)


# %%
roi = rois[0]
ref = refs[0]
med = meds[0]

fig, ax  = plt.subplots(figsize=(10, 10))
ax.imshow(ref)

buf = io.BytesIO()
fig.savefig(buf)
img1 = imageio.imread(buf)

for r, m in zip(roi, med):
    # l = r[r > 1e-8].mean()
    r = r.copy()
    r[r == 0] = np.nan
    ax.imshow(r, cmap='Reds')
    ax.scatter(m[0], m[1], c='w', marker='x')

buf = io.BytesIO()
fig.savefig(buf)
img2 = imageio.imread(buf)

# %%
with imageio.get_writer('test.gif', mode='I', duration=1000 / 2, loop=0) as writer:
    for img in [img1, img2]:
        writer.append_data(img)

# %%
