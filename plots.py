import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import matplotlib.colors as colors
import matplotlib.ticker as ticker
import matplotlib

plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"

font = {'size'   : 16}

matplotlib.rc('font', **font)

def save_histogram(x,y,df,x_bins=10000,log=False,rcubed=False):
    suffix = ""
    if rcubed:
        df[y] = df[y]*(df[x])**2
        suffix += "_rcubed"
    if log == True:
        suffix += "_log"
    fig,ax = plt.subplots()
    hist_df = df[df[x] != 0]
    x_sorted = list(hist_df[x].sort_values())
    x_bins_i = [int(i) for i in np.linspace(0,len(x_sorted)-1,x_bins)]
    x_bins = np.array([x_sorted[i] for i in x_bins_i])
    hist_unnormed,x_edge,y_edge = np.histogram2d(hist_df[x],hist_df[y],bins=[x_bins[1:-20],np.logspace(np.log10(10e-9),np.log10(1.0), 500)])
    x_sums = hist_unnormed.sum(axis=1)+10e-8
    hist_normed = hist_unnormed/x_sums[:,np.newaxis]
    norm = colors.LogNorm(vmin=0.0001,vmax=1) if log else None
    plt.pcolormesh(x_edge,y_edge,hist_normed.T,cmap="viridis",norm=norm)
    plt.colorbar()
    ax.set_yscale("log")
    if x == "x1":
        ax.set_xlabel("Atomic Seperation (Å)")
    if x == "x2":
        ax.set_xlabel("Logarithm of sine coulomb potential strength")
    ax.set_ylabel("Attention Coefficient (0-1)")
    fig.savefig("histograms/"+x+"_"+y+suffix+".png",bbox_inches='tight')
    plt.clf()
    fig,ax = plt.subplots()
    logbins = np.logspace(-8,2,100)
    ax.hist(df[df[x] == 0][y],bins=logbins)
    ax.set_xlim(10e-8,2)
    ax.set_xscale("log")
    ax.set_xlabel("Attention Coefficients where distance = 0")
    ax.set_ylabel("Frequency")
    if not rcubed:
        fig.savefig("histograms/"+x+"_"+y+suffix+"_selfloops.png",bbox_inches='tight')
    plt.clf()

def save_scatter(x,y,df):
    fig,ax = plt.subplots()
    plt.scatter(df[x],df[y])
    fig.savefig("histograms/scatter_"+x+"_"+y+".png",bbox_inches='tight')

Parity = pd.DataFrame(pd.read_csv("parity plot data.csv",index_col=0).to_numpy(),columns=["Pred","True","MAE"])
fig,ax = plt.subplots()
ax.tick_params(length=15)
ax.tick_params(which="minor",length=6)
ax.set_xlabel("true band gap (eV)")
ax.set_ylabel("predicted band gap (eV)")
ax.set_xlim([0,11])
ax.set_ylim([0,11])
ax.xaxis.set_major_locator(plt.MaxNLocator(5))
ax.yaxis.set_major_locator(plt.MaxNLocator(5))
ax.xaxis.set_minor_locator(plt.MaxNLocator(10))
ax.yaxis.set_minor_locator(plt.MaxNLocator(10))
norm = colors.LogNorm(vmin=1,vmax=10e2)
plt.hist2d(Parity["True"],Parity["Pred"],bins=100,norm=norm)
plt.colorbar(pad=-0.05)
plt.axis('square')
#plt.plot([-1,100],[-1,100])
plt.savefig("histograms/TrueVPred.png",bbox_inches='tight',dpi=600)
plt.clf()

hist_unnormed,x_edge,y_edge = np.histogram2d(Parity["True"],Parity["Pred"],100)
x_sums = hist_unnormed.sum(axis=1)+10e-8
hist_normed = hist_unnormed/x_sums[:,np.newaxis]
norm = colors.LogNorm(vmin=0.001,vmax=1)
plt.pcolormesh(x_edge,y_edge,hist_normed.T,cmap="viridis",norm=norm)
plt.colorbar()
ax.tick_params(length=15)
ax.tick_params(which="minor",length=6)
ax.set_yscale("log")
ax.set_xlabel("True Band Gap (ev)")
ax.set_ylabel("Predicted Band Gap (ev)")
fig.savefig("histograms/TrueVPred_xnorm.png",bbox_inches='tight')
plt.clf()

plt.hist(Parity[Parity["True"] == 0]["Pred"],bins=100)
plt.savefig("histograms/Pred_at_0.png",bbox_inches='tight')
plt.clf()

plt.hist(Parity[Parity["Pred"] == 0]["True"],bins=100)
plt.savefig("histograms/True_at_0.png",bbox_inches='tight')
plt.clf()

fig,ax = plt.subplots()
ax.tick_params(length=15)
ax.tick_params(which="minor",length=6)
ax.set_xlabel("MAE (ev)")
plt.hist(Parity["MAE"],bins=100)
plt.gca().set_xlim(left=0)
plt.savefig("histograms/MAE.png",bbox_inches='tight')
plt.clf()


#Loss vs epoch requires importing csvs from tensorboard and is disabled until requested, place the csvs for the run from tensorboard into a folder called "Loss_Epoch" to use

# fig,ax = plt.subplots()
# ax.tick_params(length=16)
# ax.tick_params(which="minor",length=6)

# import glob
# all_files = glob.glob("Loss_Epoch/*.csv")
# li = []

# for filename in all_files:
#     df = pd.read_csv(filename)
#     li.append(df)

# Loss_Epoch = pd.concat(li, axis=0, ignore_index=True)
# Loss_Epoch = Loss_Epoch.sort_values("Step")

# import math
# def set_aspect_ratio_log(plot, aspect_ratio):
#         x_min, x_max = plot.get_xlim()
#         y_min, y_max = plot.get_ylim()
#         return plot.set_aspect(aspect_ratio * ((math.log10(x_max / x_min)) / (math.log10(y_max / y_min))))

# ax.set_yscale("log")
# ax.set_xscale("log")
# ax.set_xlabel("number of epochs")
# ax.set_ylabel("validation MAE on band gap (eV)")
# ax.tick_params(length=15)
# ax.tick_params(which="minor",length=6)
# ax.yaxis.set_minor_formatter(ticker.ScalarFormatter())
# Loss_Epoch["Step"] = Loss_Epoch["Step"]/5259
# ax.scatter(Loss_Epoch["Step"],Loss_Epoch["Value"],s=1)
# ax.set_yticks([i/10 for i in range(2,11)],[str(i/10) for i in range(2,11)],minor=False)
# ax.set_yticks([i/100 for i in range(20,110,2)],[None for i in range(20,110,2)],minor=True)
# set_aspect_ratio_log(ax,1)
# ax.set_ylim([0.2,1])
# plt.savefig("histograms/Loss_Epoch_log_log.png",bbox_inches='tight',dpi=600)
# plt.clf()

# fig,ax = plt.subplots()  
# ax.set_xlabel("Mini-Batches")
# ax.set_ylabel("validation MAE on band gap (ev)")
# ax.tick_params(length=15)
# ax.tick_params(which="minor",length=6)
# ax.yaxis.set_minor_formatter(ticker.ScalarFormatter())
# ax.scatter(Loss_Epoch["Step"],Loss_Epoch["Value"],s=1)
# ax.axis('square')
# plt.savefig("histograms/Loss_Epoch.png",bbox_inches='tight')
# plt.clf()

chunk_size = 10e5
df = pd.concat(tqdm(pd.read_csv("attention_logs.csv",chunksize=chunk_size)))
df = df.loc[~(df.isin([0,1])).all(axis=1)]
df = df.reset_index()

fig,ax = plt.subplots(2, 3, sharex=True, sharey=True,gridspec_kw={'wspace':0.1, 'hspace':0.1},figsize=(12,8))
for block in range(1,3):
    for head in range(1,4):
        x_bins=10000
        x = "x1"
        log = True
        y = "y" + str(block) + str(head)
        print(y)
        hist_df = df[df[x] != 0]
        x_sorted = list(hist_df[x].sort_values())
        print(x_bins)
        print(len(x_sorted))
        x_bins_i = [int(i) for i in np.linspace(0,len(x_sorted)-1,x_bins)]
        x_bins = np.array([x_sorted[i] for i in x_bins_i])
        hist_unnormed,x_edge,y_edge = np.histogram2d(hist_df[x],hist_df[y],bins=[x_bins[40:],np.logspace(np.log10(10e-7),np.log10(1.0), 500)])
        x_sums = hist_unnormed.sum(axis=1)+10e-8
        hist_normed = hist_unnormed/x_sums[:,np.newaxis]
        norm = colors.LogNorm(vmin=0.0001,vmax=1)
        #ax[block-1,head-1].set(aspect='equal')
        mesh = ax[block-1,head-1].pcolormesh(x_edge,y_edge,hist_normed.T,cmap="viridis",norm=norm)
        ax[block-1,head-1].set_ylim(10**-6,1)
        ax[block-1,head-1].set_yscale("log")
        ax[block-1,head-1].set_xlim(0,15)
        ax[block-1,head-1].xaxis.set_major_locator(plt.MaxNLocator(3))
        ax[block-1,head-1].xaxis.set_minor_locator(plt.MaxNLocator(6))
        ax[block-1,head-1].set_yticks([10**-6,10**-4,10**-2,1],["$10^{-6}$","$10^{-4}$","$10^{-2}$","$10^{-0}$"],minor=False)
        ax[block-1,head-1].set_yticks([10**-5,10**-3,10**-1],[None,None,None],minor=True)
        ax[block-1,head-1].tick_params(length=12)
        ax[block-1,head-1].tick_params(which="minor",length=6)
ax[0,0].set_xlabel("Attention head 1")
ax[0,0].xaxis.set_label_position('top')
ax[0,0].xaxis.labelpad = 15
ax[0,1].set_xlabel("Attention head 2")
ax[0,1].xaxis.set_label_position('top')
ax[0,1].xaxis.labelpad = 15
ax[0,2].set_xlabel("Attention head 3")
ax[0,2].xaxis.set_label_position('top')
ax[0,2].xaxis.labelpad = 15
ax[0,0].set_ylabel("Attention block 1")
ax[0,0].yaxis.labelpad = 45
ax[1,0].set_ylabel("Attention block 2")
ax[1,0].yaxis.labelpad = 45
shadowaxes = fig.add_subplot(111, xticks=[], yticks=[], frame_on=False)
shadowaxes.yaxis.labelpad = 55
shadowaxes.xaxis.labelpad = 25
#shadowaxes.xaxis.label.set_size(22)
#shadowaxes.yaxis.label.set_size(22)
shadowaxes.set_ylabel("attention weight")
shadowaxes.set_xlabel("interatomic distance (Å)")
cbar_ax = fig.add_axes([0.93, 0.11, 0.02, 0.77])
cb = fig.colorbar(mesh,cax=cbar_ax)
cb.ax.tick_params(length=10)
cb.ax.tick_params(which="minor",length=6)

#plt.grid(True)
plt.savefig("histograms/distance_coefficients.png",bbox_inches='tight')
fig,ax = plt.subplots()                
ax.hist(df["x1"],bins=np.linspace(0,30,301))
ax.set_xlabel("Atomic Seperation (Angstrom)")
plt.savefig("histograms/histograms_x1.png",bbox_inches='tight')
plt.clf()
fig,ax = plt.subplots()
ax.hist(df["x2"],bins=np.linspace(-11,11,301))
ax.set_xlabel("Logarithm of sine coulomb potential strength")
plt.savefig("histograms/histograms_x2.png",bbox_inches='tight')