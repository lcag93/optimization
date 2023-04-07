
import os
import numpy as np
import pandas as pd
import xarray as xr


import matplotlib.pyplot as plt
from matplotlib import gridspec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib as mpl

# import cartopy
# import cartopy.crs as ccrs
# from cartopy.feature import ShapelyFeature
# from cartopy.io.shapereader import Reader
# import cartopy.feature as cfeature



def axplot_scatter_mda_vs_data(ax, x_mda, y_mda, x_data, y_data):
    'axes scatter plot variable1 vs variable2 mda vs data'

    # full dataset 
    ax.scatter(
        x_data, y_data,
        marker = '.',
        c = 'thistle', alpha=0.7,
        s = 2, label = 'dataset'
    )


    # mda selection
    p2 = ax.scatter(
        x_mda, y_mda,
        marker = '.',
        c = range(len(x_mda)),cmap='plasma_r',
        s = 50, label='subset'
            )

    return p2

def Plot_Data(pd_data, d_lab, color=[], figsize=[13,12]):
    '''
    Plot scatter with MDA selection vs original data

    pd_data - pandas.DataFrame, complete data
    pd_mda - pandas.DataFrame, mda selected data

    pd_data and pd_mda should share columns names
    '''

    # variables to plot
    vns = pd_data.columns

    # filter
    vfs = ['n_sim']
    vns = [v for v in vns if v not in vfs]
    n = len(vns)

    # figure
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(n-1, n-1, wspace=0.001, hspace=0.001)

    for i in range(n):
        for j in range(i+1, n):

            # get variables to plot
            vn1 = vns[i]
            vn2 = vns[j]

            # mda Entire-dataset
            vv1_dat = pd_data[vn1].values[:]
            vv2_dat = pd_data[vn2].values[:]

            # scatter plot 
            ax = plt.subplot(gs[i, j-1])
            if color:
                im = ax.scatter(vv2_dat, vv1_dat,c = color,
                        s=6, cmap = 'magma_r', alpha=0.8,
                        label = 'dataset')
                plt.colorbar(im).set_label('V (m3)', fontsize=15, color='purple')
            else:
                im = ax.scatter(vv2_dat, vv1_dat,c = 'plum',
                        s=3, cmap = 'magma_r', alpha=0.7,
                        label = 'dataset') 

            # custom axes
            if j==i+1:
                ax.set_xlabel(d_lab[vn2],{'fontsize':12, 'fontweight':'bold'})
            if j==i+1:
                ax.set_ylabel(d_lab[vn1],{'fontsize':12, 'fontweight':'bold'})

            if i==0 and j==n-1:
                ax.legend(fontsize=14)
                
def Plot_MDA(pd_mda, d_lab, color=[], m_s=6, figsize=[11,9], label='Sel. Order', cmap='magma_r'):
    '''
    Plot scatter with MDA selection 

    pd_mda - pandas.DataFrame, mda selected data

    d_lab: column labels
    
    '''

    # variables to plot
    vns = pd_mda.columns

    # filter
    vfs = ['n_sim']
    vns = [v for v in vns if v not in vfs]
    n = len(vns)

    # figure
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(n-1, n-1, wspace=0.001, hspace=0.001)

    for i in range(n):
        for j in range(i+1, n):

            # get variables to plot
            vn1 = vns[i]
            vn2 = vns[j]

            # mda and entire-dataset
            vv1_mda = pd_mda[vn1].values[:]
            vv2_mda = pd_mda[vn2].values[:]

            # scatter plot 
            ax = plt.subplot(gs[i, j-1])
            
            if len(color):
                im = ax.scatter(vv2_mda, vv1_mda,c = color,
                            s=m_s, cmap = cmap, alpha=0.7,
                            label = 'dataset')
            else:
                im = ax.scatter(vv2_mda, vv1_mda,c = range(np.shape(pd_mda)[0]),
                            s=m_s, cmap = cmap, alpha=0.7,
                            label = 'dataset')

            # custom axes
            if j==i+1:
                ax.set_xlabel( d_lab[vn2],{'fontsize':16, 'fontweight':'bold'})
            if j==i+1:
                ax.set_ylabel(d_lab[vn1],{'fontsize':16, 'fontweight':'bold'})

            if i==0 and j==n-1:
                ax.legend(fontsize=14)
                
    gs.tight_layout(fig, rect=[0.05, 0.01, 0.91, 0.99])   
    
    gs2=gridspec.GridSpec(1,1)
    ax1=fig.add_subplot(gs2[0])
    plt.colorbar(im,cax=ax1)
    ax1.set_ylabel(label)
    gs2.tight_layout(fig, rect=[0.91, 0.05, 1, 0.33])
    
def Plot_MDA_Data(pd_data, pd_mda, d_lab, figsize=[13,12]):
    '''
    Plot scatter with MDA selection vs original data

    pd_data - pandas.DataFrame, complete data
    pd_mda - pandas.DataFrame, mda selected data

    pd_data and pd_mda should share columns names
    '''

    # variables to plot
    vns = pd_data.columns

    # filter
    vfs = ['n_sim']
    vns = [v for v in vns if v not in vfs]
    n = len(vns)

    # figure
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(n-1, n-1, wspace=0.001, hspace=0.001)

    for i in range(n):
        for j in range(i+1, n):

            # get variables to plot
            vn1 = vns[i]
            vn2 = vns[j]

            # mda and entire-dataset
            vv1_mda = pd_mda[vn1].values[:]
            vv2_mda = pd_mda[vn2].values[:]

            vv1_dat = pd_data[vn1].values[:]
            vv2_dat = pd_data[vn2].values[:]

            # scatter plot 
            ax = plt.subplot(gs[i, j-1])
            im = axplot_scatter_mda_vs_data(ax, vv2_mda, vv1_mda, vv2_dat, vv1_dat)

            # custom axes
            if j==i+1:
                ax.set_xlabel( d_lab[vn2],{'fontsize':12, 'fontweight':'bold'})
            if j==i+1:
                ax.set_ylabel(d_lab[vn1],{'fontsize':12, 'fontweight':'bold'})

            if i==0 and j==n-1:
                ax.legend(fontsize=14)
                
    gs.tight_layout(fig, rect=[0.05, 0.01, 0.91, 0.99])   
    
    gs2=gridspec.GridSpec(1,1)
    ax1=fig.add_subplot(gs2[0])
    plt.colorbar(im,cax=ax1)
    ax1.set_ylabel('Sel. Order')
    gs2.tight_layout(fig, rect=[0.91, 0.05, 1, 0.33])
    


def PC_EOF_plot(sp_fields, cini, cend, figsize=[20,5]):
    
    for c in np.arange(cini, cend):
        
        fig = plt.figure(figsize=figsize)
        gs = gridspec.GridSpec(1,3, wspace=0.2, hspace=0.001)
        ax=fig.add_subplot(gs[0],projection = ccrs.PlateCarree(central_longitude=180))
        
        z=np.reshape(sp_fields.EOFs.values[c,:],(len(sp_fields.longitude),len(sp_fields.latitude)))
        a1=ax.contourf(sp_fields.longitude, sp_fields.latitude, z.T, transform=ccrs.PlateCarree(), cmap='RdBu_r', vmin=-.7, vmax=.7)
        ax.add_feature(cfeature.LAND, color='lightgrey', zorder=2)
        ax.set_extent([sp_fields.longitude[0], sp_fields.longitude[-1], sp_fields.latitude[0], sp_fields.latitude[-1]])
        cbaxes = inset_axes(ax, width="96%", height="3%", loc=3) 
        plt.colorbar(a1, cax=cbaxes, orientation='horizontal', shrink=0.6)
        
        ax=fig.add_subplot(gs[1:3])
        ax.plot(sp_fields.time, sp_fields.PCs.isel(pc=c).values, color='navy')
        ax.set_xlim([sp_fields.time[0], sp_fields.time[-1]])
        ax.grid(color='royalblue', alpha=0.2)
        ax.set_xlabel('Time', fontsize=15, color='navy')
        ax.set_ylabel('PC ' + str(c), fontsize=15, color='navy', labelpad=0.005)
        ax.set_ylim([-np.nanmax(sp_fields.PCs)-1, np.nanmax(sp_fields.PCs)+1])
        
        
        
# def RBF_output_plot(sp_fields, output, cini, cend, var='SST', figsize=[10,10], cmap='RdBu_r'):
    
#     n=int(np.ceil(np.sqrt(cend-cini)))
#     fig = plt.figure(figsize=figsize)
#     gs = gridspec.GridSpec(n,n, wspace=0.005, hspace=0.005)
    
#     Z_ALL=[]
    
#     for c in np.arange(cini, cend):   
        
#         ax=fig.add_subplot(gs[c],projection = ccrs.PlateCarree(central_longitude=180))
        
#         Z=np.zeros((len(sp_fields.longitude),len(sp_fields.latitude)))
#         for s in range(sp_fields.n_pcs.values):
#             z=np.reshape(sp_fields.EOFs.values[s,:],(len(sp_fields.longitude),len(sp_fields.latitude)))
#             Z+=z*output[c,s]
#         ext=np.nanmax([np.nanmax(Z), np.abs(np.nanmin(Z))])            
#         a1=ax.contourf(sp_fields.longitude, sp_fields.latitude, Z.T, transform=ccrs.PlateCarree(), cmap=cmap, vmin=-ext, vmax=ext)
#         ax.add_feature(cfeature.LAND, color='lightgrey', zorder=2)
#         ax.set_extent([sp_fields.longitude[0], sp_fields.longitude[-1], sp_fields.latitude[0], sp_fields.latitude[-1]])
#         Z_ALL.append(Z)
        
#     gs.tight_layout(fig, rect=[0.05, 0.15, 1, 1]) 
    
#     gs2=gridspec.GridSpec(1,1)
#     ax1=fig.add_subplot(gs2[0])
#     plt.colorbar(a1,cax=ax1, orientation='horizontal')
    
#     cmap = mpl.cm.RdBu_r
#     norm = mpl.colors.Normalize(vmin=-ext, vmax=ext)

#     cb1 = mpl.colorbar.ColorbarBase(ax1, cmap=cmap,
#                                     norm=norm,
#                                     orientation='horizontal')
#     cb1.set_label(var, fontsize=15)

# #     plt.colorbar(a1,cax=ax1, orientation='horizontal')
    
# #     ax1.set_ylabel('SST')
#     gs2.tight_layout(fig, rect=[0.2, 0.05, 0.8, 0.15])
    
#     Z_ALL = xr.Dataset({var: (['time','lat','lon'],Z_ALL)    
#                    }, 
#                   coords={'case': np.arange(cini, cend), 
#                           'longiture': sp_fields.longitude.values, 
#                           'latitude': sp_fields.latitude.values})
    
#     return Z_ALL


# def RBF_var_reconstruct(sp_fields, output, cini, cend, var='SST'):    
    
#     Z_ALL=[]
    
#     for c in np.arange(cini, cend):   
#         Z=np.zeros((len(sp_fields.longitude),len(sp_fields.latitude)))
#         for s in range(sp_fields.n_pcs.values):
#             z=np.reshape(sp_fields.EOFs.values[s,:],(len(sp_fields.longitude),len(sp_fields.latitude)))
#             Z+=z*output[c,s]
#         Z+=np.reshape(sp_fields.means.values,(len(sp_fields.longitude),len(sp_fields.latitude)))
#         Z_ALL.append(Z)
        
#     Z_ALL = xr.Dataset({var: (['time','lat','lon'],Z_ALL)    
#                    }, 
#                   coords={'case': np.arange(cini, cend), 
#                           'longitude': sp_fields.longitude.values, 
#                           'latitude': sp_fields.latitude.values})
    
#     return Z_ALL


def RBF_var_reconstruct(C_R, output, cini, cend, var='SST'):    
    
    Z_ALL=[]
    
    for c in np.arange(cini, cend):   
        Z=np.zeros((len(C_R.ny),len(C_R.nx)))
        for s in range(C_R.n_pcs.values):
            z=np.reshape(C_R.EOFs.values[s,:],(len(C_R.ny),len(C_R.nx)))
            ss = z*output[c,s]
            ss = np.where(np.isnan(ss),0,ss)
            Z+=ss
        Z+=np.reshape(C_R.means.values,(len(C_R.ny),len(C_R.nx)))
        Z_ALL.append(Z)
        
    Z_ALL = xr.Dataset({var: (['index','ny','nx'],Z_ALL)    
                   }, 
                  coords={'index': np.arange(cini, cend), 
                          'nx': C_R.nx.values, 
                          'ny': C_R.ny.values})
    
    return Z_ALL


# def RBF_output_plot(Z_ALL, cini, cend, var='SST', figsize=[10,10]):
    
#     n=int(np.ceil(np.sqrt(cend-cini)))
#     fig = plt.figure(figsize=figsize)
#     gs = gridspec.GridSpec(n,n, wspace=0.005, hspace=0.005)
#     cont=0    
#     for c in np.arange(cini, cend):   
        
#         ax=fig.add_subplot(gs[cont],projection = ccrs.PlateCarree(central_longitude=180))
#         cont+=1 
#         a1=ax.contourf(Z_ALL.longitude, Z_ALL.latitude, Z_ALL[var].isel(time=c).T, transform=ccrs.PlateCarree(), 
#                        cmap='turbo', vmin=np.nanmin(Z_ALL[var].values[cini:cend,:,:]), vmax=np.nanmax(Z_ALL[var].values[cini:cend,:,:]))
#         ax.add_feature(cfeature.LAND, color='lightgrey', zorder=2)
#         ax.set_extent([Z_ALL.longitude[0], Z_ALL.longitude[-1], Z_ALL.latitude[0], Z_ALL.latitude[-1]])
        
#     gs.tight_layout(fig, rect=[0.05, 0.15, 1, 1]) 
    
#     gs2=gridspec.GridSpec(1,1)
#     ax1=fig.add_subplot(gs2[0])
#     plt.colorbar(a1,cax=ax1, orientation='horizontal')
    
#     cmap = mpl.cm.turbo
#     norm = mpl.colors.Normalize(vmin=np.nanmin(Z_ALL[var].values[cini:cend,:,:]), vmax=np.nanmax(Z_ALL[var].values[cini:cend,:,:]))

#     cb1 = mpl.colorbar.ColorbarBase(ax1, cmap=cmap,
#                                     norm=norm,
#                                     orientation='horizontal')
#     cb1.set_label(var, fontsize=15)

#     gs2.tight_layout(fig, rect=[0.2, 0.05, 0.8, 0.15])


def Plot_Differences(xds_var, Z, vmax=0.2, ax=[], xlim=[], ylim=[], p_coast=1):
    
    if not ax:
        fig = plt.figure(figsize=figsize)
        gs = gridspec.GridSpec(1,1, hspace = 0.005, wspace = 0.005)
        ax = fig.add_subplot(gs[0,0])

    xa, ya = 'globalx', 'globaly'
    X = xds_var[xa].values[:]
    Y = xds_var[ya].values[:]


    pm = ax.pcolormesh(
                    X,Y,Z,
                    cmap='RdBu_r',
                    vmin=-vmax, vmax=vmax,
                    alpha=1,
                )    
    if p_coast:
        try:
            Z = xds_var['zb'].values[0,:,:]
        except:
            Z = xds_var['zb'].values
        ax.contour(X, Y, Z, colors='black', levels=[0])
    
    if xlim:
        ax.set_xlim(xlim[0],xlim[1])
    else:
        ax.set_xlim([np.nanmin(X), np.nanmax(X)])
    if ylim:
        ax.set_ylim(ylim[0],ylim[1])
    else:
        ax.set_ylim([np.nanmin(Y), np.nanmax(Y)])
    
    ax.set_aspect('equal')
    ax.axis('off')
    label='Differences'

    plt.colorbar(pm, shrink=.4, orientation='vertical', pad=0.02).set_label(label, fontsize=16)
