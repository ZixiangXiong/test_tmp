# -*- coding: utf-8 -*-

# EnKF cycling for SQG turbulence model model with boundary temp obs,
# horizontal and vertical localization.  Relaxation to prior spread
# inflation, or Hodyss and Campbell inflation.
# Random or fixed observing network (obs on either boundary or
# both).

#############
# Libraries #
#############
from __future__ import print_function
import matplotlib

matplotlib.use('Agg')
from sqgturb import SQG, rfft2, irfft2
import numpy as np
from netCDF4 import Dataset
import sys, time, os
from enkf_utils import cartdist, enkf_update, gaspcohn
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from Rev_SDE import REVERSE_SDE
import pandas as pd
import math

matplotlib.rcParams.update({'font.size': 16.0})


def main():
    ##############
    # User input #
    ##############

    ## NWP/EnKF
    N1 = 64  #  number of points in the x and y direction (on one of the PV levels)
    nanals = 20  # number of ensemble members
    nassim = 400  # assimilation times to run (orig=1600)
    nassim_spinup = 200  # diagnostics will be done after the (nassim_spinup+1)th cycle (orig=800)
    nassim_write = 0  #  data will be written out only after the (numassim_write+1)th cycle
    (use_ensf, use_letkf) = (False, True)  # if False, use serial EnSRF filter (orig=False)     ##1234567890
    rmse_file = np.zeros((nassim, 2))  ##1234567890
    rmse_file[:, 0] = np.arange(nassim, dtype=int)  ##1234567890

    ## Observations
    # number of obs to assimilate (orig=1024)                                                   ##1234567890
    # nobs = -2   ## the actual obs-number is (64/2)**2=1024                                      ##1234567890
    # nobs = 1024                                                                                  ##1234567890
    nobs = 4096
    # nobs = 32
    # nobs = 128
    # nobs = 64
    # nobs = 16
    # nobs = 512
    # nobs = 2048
    # nobs = -4   ## the actual obs-number is (64/4)**2=256                                          ##1234567890
    # nobs = -4                                                                                       ##1234567890
    # nobs = 256                                                                                      ##1234567890
    #   if nobs > 0: observations randomly sampled (without replacement) during each cycle (RANDOM)
    #    if nobs = -p: fixed network - observations placed every "p" grid points (FIXED_EVEN)
    #   if nobs = -1: fixed network - observations at all grid points
    hybrid_network = 1  # if 1, network is randomly generated but remains the same at all cycles (FIXED)
    percentage_arctan = 1.  # if >0, it means the observation is mixed with linear and nonlinear.
    oberrstdev = 1.0  # observation error standard deviation in K (orig=1)
    oberrstdev_nl = 0.01
    ## Input arguments
    #hcovlocal_scale = 1200.0 * 1000.0  #  [m]
    hcovlocal_scale = 2800.0 * 1000.0  #  [m]
    # covinflate1 = 0.5
    covinflate1 = 0.8
    covinflate2 = -1.0
    ## Fix the random seed
    rns1 = np.random.RandomState(42)
    rns2 = np.random.RandomState(42)
    # if levob=0, sfc temp obs used.  if 1, lid temp obs used. If [0,1] obs at both
    # boundaries.
    levob = [0, 1];
    levob = list(levob);
    levob.sort()  # which levels are obs to be assimilated (orig=[0,1])
    # if levob=0: use obs at lower boundary
    # if levob=1: use obs at upper boundary
    # if levob=[0,1]: use obs at both levels
    direct_insertion = False  # only relevant for nobs=-1, levob=[0,1] (orig=False)

    ## I/O paths
    # savedata = True  # save data (orig=False)
    savedata = False  ##1234567890
    data_dir = r"/Users/xiongzixiang/PycharmProjects/EnSF_banking"
    filename_climo = '{}/sqg_N{}_12hrly.nc'.format(data_dir, N1)  # file name for forecast model climatology
    fname_ncout = r'C:\Users\Zixiang\PycharmProjects\Research\sqg_enkf_p{}_RANDOM.nc'.format(nobs)
    if nobs < 0:
        nobs_fixed = int((-N1 / nobs) ** 2)
        fname_ncout = r'C:\Users\Zixiang\PycharmProjects\Research\sqg_enkf_p{}_FIXED.nc'.format(
            nobs_fixed)
    if hybrid_network == 1:
        fname_ncout = r'C:\Users\Zixiang\PycharmProjects\Research\sqg_enkf_p{}_FIXED_EVEN.nc'.format(
            nobs)
    filename_truth = '{}/sqg_N{}_12hrly.nc'.format(data_dir, N1)  # file name for nature run to draw observations
    outdir_figs = r'C:\Users\Zixiang\PycharmProjects\Research\observation_density_tunedParms'

    ## Plotting
    plot_KE_spectra_error_spread = True
    plot_ob_network = False
    plot_cov_localiation = False

    ## Misc
    profile_cpu = True  # CPU profiling for analysis and forecast steps

    ## Print info
    print("                                         *** STARTING SQG CYCLING ***")
    print("\nHorizontal localization scale = {:.1f} km".format(int(hcovlocal_scale) / 1000.))
    print("Number of model grid points (N1) = {}".format(N1))
    if covinflate2 == -1.0:
        infl_method = 'rtps'
        print("Will be using RTPS inflation with a factor of {:.1f}".format(covinflate1))
    else:
        infl_method = 'hodyss'
        print("Will be using Hodyss inflation with a={:.1f} and b={:.1f}".format(covinflate1, covinflate2))

    ################
    # Main program #
    ################
    print("\n\n\n=== Set things up before cycling ===\n")


    ## Assign important model variables
    print("* Assign important model variables")
    nc_climo = Dataset(filename_climo)
    scalefact = nc_climo.f * nc_climo.theta0 / nc_climo.g  #  PV-T conversion factor so that du/dz=d(pv)/dy
    x = nc_climo.variables['x'][:]
    y = nc_climo.variables['y'][:]
    x, y = np.meshgrid(x, y)
    nx = len(x)
    ny = len(y)
    dt = nc_climo.dt
    diff_efold = nc_climo.diff_efold

    ## Read in model climatology
    pv_climo = nc_climo.variables['pv']  #  PV fields from all times during the
    # free forecast run

    ## Initialize background ensemble by picking PV fields from randomly selected
    ## climatology times
    print("* Initialize model ensemble from model climatology")
    pvens = np.empty((nanals, 2, ny, nx), np.float32)
    indxran = rns2.choice(pv_climo.shape[0], size=nanals, \
                               replace=False)  # set of randomly selected time indices
    threads = int(os.getenv('OMP_NUM_THREADS', '1'))
    models = []  #  list to hold all ensemble member instances
    for nanal in range(nanals):
        pvens[nanal] = pv_climo[indxran[nanal]]
        if use_ensf is True:  ##1234567890
            pvens[nanal] = pv_climo[0] + rns1.normal(0., 1000., size=(2, ny, nx))  ##1234567890
        models.append( \
            SQG(pvens[nanal],
                nsq=nc_climo.nsq, f=nc_climo.f, dt=dt, U=nc_climo.U, H=nc_climo.H, \
                r=nc_climo.r, tdiab=nc_climo.tdiab, symmetric=nc_climo.symmetric, \
                diff_order=nc_climo.diff_order, diff_efold=diff_efold, threads=threads))

    ## True PV field at all times
    print("* Assign the true PV field at all times")
    nc_truth = Dataset(filename_truth)
    pv_truth = nc_truth.variables['pv']

    ## Vertical localization
    Lr = np.sqrt(models[0].nsq) * models[0].H / models[0].f  # Rossby radius of deformation
    vcovlocal_fact = gaspcohn(np.array(Lr / hcovlocal_scale))
    vcovlocal_fact = float(vcovlocal_fact)
    print("* Vertical localization factor: {:.1f}".format(vcovlocal_fact))

    ## Set up some arrays that will be used during DA
    print("* Set up some arrays that will be used during DA")
    if nobs < 0:  #  fixed observation network
        nskip = -nobs
        if nx % nobs != 0:
            raise ValueError('  Error: nx must be divisible by nobs')
        nobs = (nx // nobs) ** 2
        print('     nobs = %s' % nobs)
        fixed = True
        obs_network_type = "Fixed_even"
    else:
        fixed = False
        if hybrid_network == 1:
            obs_network_type = "Fixed"
        else:
            obs_network_type = "Random"
    oberrvar = oberrstdev ** 2 * np.ones(nobs, np.float32)  # 1D array of ob variances [p].
    oberrvar_mean = oberrvar.mean()  # mean observation error variance
    pvob = np.empty((len(levob), nobs), np.float32)  #  2D array of observations [lev,p]
    covlocal = np.empty((ny, nx), np.float32)  # 2D array of covariance functions [Nx,Ny]
    covlocal_tmp = np.empty((nobs, nx * ny), np.float32)  # 2D array of covariance functions [p,Nx*Ny]
    xens = np.empty((nanals, 2, nx * ny), np.float32)  # state vector [K,levs,Nx*Ny]
    if not use_letkf:  # if we use EnSRF, will use an observation localization function
        obcovlocal = np.empty((nobs, nobs), np.float32)  #  2D array of size [p,p]
    else:
        obcovlocal = None
    obtimes = nc_truth.variables['t'][:]
    assim_interval = obtimes[1] - obtimes[0]
    assim_timesteps = int(np.round(assim_interval / models[0].dt))  #  num time steps from one
    # analysis time to the next

    ## Specify cycle starting time and forecast integration length
    print("* Specify cycle starting time and forecast integration length")
    for nanal in range(nanals):
        models[nanal].t = obtimes[0]
        models[nanal].timesteps = assim_timesteps

    ## Declare some variables needed for the KE spectra error and spread
    print("* Declare some variables needed for the KE spectra error and spread")
    kespec_errmean = None  #  KE spectra error mean
    kespec_sprdmean = None  # KE spectra spread
    ncount_spinup = 0  #  number of DA cycles over which KE spectra statistics will be computed
    nanals2 = 4  # ensemble members used for KE spectra spread calculation

    ###########
    # Cycling #
    ###########
    print("\n\n=== Cycling ===")
    for ntime in range(nassim):
        # for ntime in range(1):
        tcycle_1 = time.time()
        cycle_num = ntime + 1
        print("\nWorking on cycle {}/{}".format(cycle_num, nassim))

        ## Initializing netCDF output file
        if savedata and (ntime == nassim_write):
            # print("     Initializing netCDF output file")
            #  open netCDF file
            nc = Dataset(fname_ncout, mode='w', format='NETCDF4')
            # specify object attributes
            nc.r = models[0].r
            nc.f = models[0].f
            nc.U = models[0].U
            nc.L = models[0].L
            nc.H = models[0].H
            nc.nanals = nanals
            nc.hcovlocal_scale = hcovlocal_scale
            nc.vcovlocal_fact = vcovlocal_fact
            nc.oberrstdev = oberrstdev
            nc.levob = levob
            nc.g = nc_climo.g
            nc.theta0 = nc_climo.theta0
            nc.nsq = models[0].nsq
            nc.tdiab = models[0].tdiab
            nc.dt = models[0].dt
            nc.diff_efold = models[0].diff_efold
            nc.diff_order = models[0].diff_order
            nc.filename_climo = filename_climo
            nc.filename_truth = filename_truth
            nc.symmetric = models[0].symmetric
            #  create dimensions
            xdim = nc.createDimension('x', models[0].N)
            ydim = nc.createDimension('y', models[0].N)
            z = nc.createDimension('z', 2)
            t = nc.createDimension('t', nassim - nassim_write)
            obs = nc.createDimension('obs', nobs)
            obs2 = nc.createDimension('obs2', len(levob) * nobs)
            ens = nc.createDimension('ens', nanals)
            # create variables
            tvar = nc.createVariable('t', np.float64, ('t',))
            ensvar = nc.createVariable('ens', np.int64, ('ens',))
            zvar = nc.createVariable('z', np.float64, ('z',))
            yvar = nc.createVariable('y', np.float64, ('y',))
            xvar = nc.createVariable('x', np.float64, ('x',))
            x_obs = nc.createVariable('x_obs', np.float64, ('t', 'obs'))
            y_obs = nc.createVariable('y_obs', np.float64, ('t', 'obs'))
            pv_obs = nc.createVariable('obs', np.float64, ('t', 'obs2'), zlib=True)
            pv_t = nc.createVariable('pv_t', np.float64, ('t', 'z', 'y', 'x'), zlib=True)
            pv_b = nc.createVariable('pv_b', np.float64, ('t', 'ens', 'z', 'y', 'x'), zlib=True)
            pv_a = nc.createVariable('pv_a', np.float64, ('t', 'ens', 'z', 'y', 'x'), zlib=True)
            inf = nc.createVariable('inflation', np.float64, ('t', 'z', 'y', 'x'), zlib=True)
            rmsi_b_nc = nc.createVariable('rmsi_b', np.float64, ('t',))
            total_spread_b_nc = nc.createVariable('total_spread_b', np.float64, ('t',))
            rmsi_a_nc = nc.createVariable('rmsi_a', np.float64, ('t',))
            total_spread_a_nc = nc.createVariable('total_spread_a', np.float64, ('t',))
            # set up units
            pv_t.units = 'K'
            pv_b.units = 'K'
            pv_a.units = 'K'
            pv_obs.units = 'K'
            xvar.units = 'meters'
            yvar.units = 'meters'
            zvar.units = 'meters'
            tvar.units = 'seconds'
            ensvar.units = 'dimensionless'
            inf.units = 'dimensionless'
            rmsi_b_nc.units = 'dimensionless'
            total_spread_b_nc.units = 'dimensionless'
            rmsi_a_nc.units = 'dimensionless'
            total_spread_a_nc.units = 'dimensionless'
            # assign coordinate variables using info from model climatology
            xvar[:] = np.arange(0, models[0].L, models[0].L / models[0].N)
            yvar[:] = np.arange(0, models[0].L, models[0].L / models[0].N)
            zvar[0] = 0;
            zvar[1] = models[0].H
            ensvar[:] = np.arange(1, nanals + 1)

        ################
        # Observations #
        ################
        # print("     Assigning observations for this cycle")

        ## Check for consistency between model and observation times
        if models[0].t != obtimes[ntime]:
            raise ValueError('      Error: Mismatch between model ({}) and observation ({}) times'. \
                             format(models[0].t, obtimes[ntime]))

        ## Determine PV obs and their {x,y} coordinates
        ## Note {x,y} coordinates refer to both levels
        if not fixed:
            if hybrid_network == 1:  ## Fixed
                np.random.seed(10)  # fix random seed so that you get the same network every time
            p = np.ones((ny, nx), np.float64) / (nx * ny)  ## Random
            indxob = np.sort(np.random.choice(nx * ny, nobs, replace=False, p=p.ravel()))
        else:  ## Fixed_even
            mask = np.zeros((ny, nx), bool)
            # if every other grid point observed, shift every other time step
            # so every grid point is observed in 2 cycle.
            """
            if nskip == 2 and ntime % 2:
                mask[1:ny:nskip, 1:nx:nskip] = True
            else:
                mask[0:ny:nskip, 0:nx:nskip] = True
            """
            if np.int_(np.sqrt(nobs)) == np.sqrt(nobs):
                nskip = int(nx / np.sqrt(nobs))
                mask[0:ny:nskip, 0:nx:nskip] = True
            else:
                nskip = int(nx / np.sqrt(nobs * 2.))
                mask[0:ny:nskip, 0:nx:(2 * nskip)] = True
            tmp = np.arange(0, nx * ny).reshape(ny, nx)
            indxob = np.sort(tmp[mask.nonzero()].ravel())


        if percentage_arctan > 0:
            indx_indxob_l = np.sort(rns2.choice(np.arange(nobs), int(nobs * (1 - percentage_arctan)), replace=False), axis=None)
            indxob_l = indxob[indx_indxob_l]
            indx_indxob_nl = np.setdiff1d(np.arange(nobs), indx_indxob_l)
            indxob_nl = indxob[indx_indxob_nl]
            oberrvar[[indx_indxob_nl]] = oberrstdev_nl ** 2
            print("indxob_l[0:10]", indxob_l[0:10], indxob_l.shape)
            print("indx_indxob_l[0:10]", indx_indxob_l[0:10], indx_indxob_l.shape)
            print("indxob_nl[0:10]", indxob_nl[0:10], indxob_nl.shape)
            print("indx_indxob_nl[0:10]", indx_indxob_nl[0:10], indx_indxob_nl.shape)
        print("indxob", indxob, indxob.shape)

        if percentage_arctan <= 0:  ## observation function is only linear.
            for k in range(len(levob)):
                # surface temp obs
                ## pv_truth[ntime, k, :, :].shape=(64,64), pv_truth[ntime, k, :, :].ravel().shape= 4096
                pvob[k] = scalefact * pv_truth[ntime, k, :, :].ravel()[indxob]
                pvob[k] += rns1.normal(scale=oberrstdev, size=nobs)  # add ob errors
        elif percentage_arctan > 0:  ## observation function is mixed of linear and nonlinear.
            for k in range(len(levob)):
                pvob[k][[indx_indxob_l]] = scalefact * pv_truth[ntime, k, :, :].ravel()[indxob_l]
                # print(scalefact * pv_truth[ntime, k, :, :].ravel()[84],pvob[k][1])
                # print(scalefact * pv_truth[ntime, k, :, :].ravel()[2595], pvob[k][2])
                # print("pvob[k][[indx_indxob_l]]",pvob[k][1],pvob[k][2],pvob[k][3],pvob[k][5])
                pvob[k][[indx_indxob_l]] += rns2.normal(loc=0., scale=oberrstdev,size=indx_indxob_l.shape[0])  # add ob errors
                # print(pvob[k][1],pvob[k][2])
                # print("pvob[k][[indx_indxob_l]]", pvob[k][1],pvob[k][2],pvob[k][3],pvob[k][5])
                pvob[k][[indx_indxob_nl]] = np.arctan(scalefact * pv_truth[ntime, k, :, :].ravel()[indxob_nl])
                # print(np.arctan(scalefact * pv_truth[ntime, k, :, :].ravel()[3159]),pvob[k][0])
                # print(np.arctan(scalefact * pv_truth[ntime, k, :, :].ravel()[2041]), pvob[k][4])
                # print("pvob[k][[indx_indxob_nl]]", pvob[k][0],pvob[k][4],pvob[k][13],pvob[k][17])
                pvob[k][[indx_indxob_nl]] += rns1.normal(loc=0., scale=oberrstdev_nl,size=indx_indxob_nl.shape[0])  # add ob errors
        xob = x.ravel()[indxob]
        yob = y.ravel()[indxob]
        pvens_copy = pvens.copy()                                                   ##1234567890
        indxob2 = np.concatenate((indxob, np.add(indxob, N1 * N1)), axis=0)  ##1234567890
        pvens_copy_obsnet = np.zeros((nanals, 2 * nobs))  ##1234567890
        for i in range(nanals):  ##1234567890
            pvens_copy_obsnet[i, :] = pvens_copy.reshape(nanals, 2 * nx * ny)[i, [indxob2]]  ##1234567890
        std_XXens_step0 = np.std(pvens_copy_obsnet, axis=0)  ##1234567890

        ## Optionally plot the observation network at first analysis time
        if (plot_ob_network == True) and (ntime == 0):
            print("     Plotting observation network @ lower level")
            fig, ax = plt.subplots(figsize=(10, 8))
            PV_truth_plot = scalefact * pv_truth[0, 0, ...]
            vmin_PV = -25.0
            vmax_PV = 25.0
            lev_inc = 0.1
            levs = np.arange(vmin_PV, vmax_PV + lev_inc, lev_inc)
            PLOT = ax.contourf(x, y, PV_truth_plot, levs, cmap="jet", extend='both')
            ax.scatter(xob, yob, color='k', s=10)
            # create and format the colorbar
            vmin_cbar = int(vmin_PV)
            vmax_cbar = int(vmax_PV)
            range_ticks = float(vmax_cbar - vmin_cbar)
            half_num_ticks = 5
            cbar_inc = range_ticks / (2 * half_num_ticks)
            ticks_cbar_lower_range = [int(-i * cbar_inc) for i in range(1, half_num_ticks + 1)]
            ticks_cbar_upper_range = [int(i * cbar_inc) for i in range(1, half_num_ticks + 1)]
            ticks_cbar = ticks_cbar_lower_range[::-1] + [0] + ticks_cbar_upper_range
            if ticks_cbar[0] < vmin_PV:  #  prevent exceeding the limit
                del ticks_cbar[0], ticks_cbar[-1]
            cbar = fig.colorbar(PLOT, ax=ax, format='%.1f')
            cbar.set_ticks(ticks_cbar)
            cbar.set_ticklabels(ticks_cbar)
            cbar.update_ticks()
            # ax.set_title("Observation network and $PV^t$: lower level \n cycle={}".\
            #             format(cycle_num),fontweight="bold",y=0.97)
            plt.axis('off')
            plt.savefig('obs_lower_lev_ncycle={}.png'.format(cycle_num), dpi=300, bbox_inches='tight')
            plt.close()
            print("     Plotting observation network @ upper level")
            fig, ax = plt.subplots(figsize=(8, 8))
            PV_truth_plot = pv_truth[0, 1, ...]
            ax.contourf(x, y, PV_truth_plot, 15, cmap="jet")
            ax.scatter(xob, yob, color='k', s=10)
            # ax.set_title("Observation network and $PV^t$: upper level \n cycle={}".\
            #            format(cycle_num),fontweight="bold",y=1.02)
            plt.axis('off')
            plt.savefig('obs_upper_lev_ncycle={}.png'.format(cycle_num), dpi=300, bbox_inches='tight')
            plt.close()
            # exit
            raise SystemExit
        ################################################
        # Covariance localization for each observation #
        ################################################
        # print("     Covariance localization")
        #print("xob", xob[0],xob[1],xob[2])
        #print("yob", yob[0], yob[1], yob[2])
        if not fixed or ntime == 0:
            for nob in range(nobs):
                dist = cartdist(xob[nob], yob[nob], x, y, nc_climo.L, nc_climo.L)
                covlocal = gaspcohn(dist / hcovlocal_scale)
                covlocal_tmp[nob] = covlocal.ravel()
                dist = cartdist(xob[nob], yob[nob], xob, yob, nc_climo.L, nc_climo.L)
                if not use_letkf: obcovlocal[nob] = gaspcohn(dist / hcovlocal_scale)

                ## Optionally plot covariance localization for first observation
                if plot_cov_localiation == True:
                    print("     Plotting covariance localization for first ob")
                    fig, ax = plt.subplots(figsize=(10, 8))
                    ax.contourf(x, y, covlocal, 15)
                    ax.set_title("Covariance localization for first observation", \
                                 fontweight="bold", y=1.02)
                    plt.axis('off')
                    plt.show()
                    raise SystemExit

        ## Compute forecast spread (model space) that will be used later
        ## to compute inflation factor
        fsprd = ((pvens - pvens.mean(axis=0)) ** 2).sum(axis=0) / (nanals - 1)

        #####################################
        # Innovation statistics: background #
        #####################################
        print("     Innovation statistics: background")

        ## Background ensemble in observation space
        hxens = np.empty((nanals, len(levob), nobs), np.float32)
        for nanal in range(nanals):
            for k in range(len(levob)):  #  2 levels
                if percentage_arctan <= 0:
                    hxens[nanal, k, ...] = scalefact * pvens[nanal, k, ...].ravel()[indxob]  # surface pv obs
                elif percentage_arctan > 0:
                    hxens[nanal][k][indx_indxob_l] = scalefact * pvens[nanal, k, ...].ravel()[
                        indxob_l]  # surface pv obs
                    hxens[nanal][k][indx_indxob_nl] = np.arctan(
                        scalefact * pvens[nanal, k, ...].ravel()[indxob_nl])  # surface pv obs
        hxensmean_b = hxens.mean(axis=0)

        ## Obervation space diagnostics
        spread_b_all = ((hxens - hxensmean_b) ** 2).sum(axis=0) / (nanals - 1)  # background spread at all ob locs
        # 2 levels times num obs @ each level
        d_b = pvob - hxensmean_b  # background innovation vector (db)
        rmsi_b = np.sqrt((d_b ** 2.0).mean())  #  background MSI (Mean Squared Innovation)
        bias_b = d_b.mean()  # mean background innovation (bias)
        spread_b = spread_b_all.mean()  #  mean background spread
        total_spread_b = np.sqrt(oberrvar_mean + spread_b)

        ## Model space diagnostics
        pvensmean_b = pvens.mean(axis=0).copy()  # background ens mean in physical space
        pverr_b = (scalefact * (pvensmean_b - pv_truth[ntime])) ** 2  # (B-T)^^2 vector
        pvsprd_b = ((scalefact * (pvensmean_b - pvens)) ** 2).sum(axis=0) / (nanals - 1)  # background spread vector
        # in model space

        ####################################
        # Writing data out before analysis #
        ####################################
        if savedata and (ntime >= nassim_write):
            print("     Writing data out before analysis")
            pv_t[ntime - nassim_write, :, :, :] = scalefact * pv_truth[ntime, :, :, :]
            pv_b[ntime - nassim_write, :, :, :, :] = scalefact * pvens[:, :, :, :]
            pv_obs[ntime - nassim_write, :] = pvob[:, :].ravel()  # concatinated over lower and upper levels
            x_obs[ntime - nassim_write, :] = xob[:]
            y_obs[ntime - nassim_write, :] = yob[:]
            rmsi_b_nc[ntime - nassim_write] = rmsi_b
            total_spread_b_nc[ntime - nassim_write] = total_spread_b

            #################
        # Analysis step #
        #################
        print("     *** Analysis update ***")
        t1 = time.time()
        xxens = np.zeros((nanals, 2, nobs))  ##1234567890

        ## Create 1D vector for each member
        for nanal in range(nanals):
            xens[nanal] = pvens[nanal].reshape((2, nx * ny))
            xxens[nanal, 0] = xens[nanal, 0, [indxob]]  ##1234567890
            xxens[nanal, 1] = xens[nanal, 1, [indxob]]  ##1234567890

        ## Update state vector, direct_insertion=False
        if direct_insertion and nobs == nx * ny and levob == [0, 1]:
            for nanal in range(nanals):
                xens[nanal] = \
                    pv_truth[ntime].reshape(2, nx * ny) + \
                    np.random.normal(scale=oberrstdev, size=(2, nx * ny)) / scalefact
            xens = xens - xens.mean(axis=0) + \
                   pv_truth[ntime].reshape(2, nx * ny) + \
                   np.random.normal(scale=oberrstdev, size=(2, nx * ny)) / scalefact
        elif use_ensf:
            XXens = xxens.reshape(nanals, 2 * nobs)
            ## Normalization for Xens
            mean_XXens = np.mean(XXens, axis=0)
            std_XXens = np.std(XXens, axis=0)
            XXens_normalization = (XXens - mean_XXens) / std_XXens

            Pvob = pvob.reshape(2 * nobs)
            ## Normalization for Pvob and obs_sigma
            Pvob_normalization = (Pvob - scalefact * mean_XXens) / std_XXens
            # Pvob_normalization = np.arctan(((np.tan(Pvob) / scalefact - mean_Xens) / std_Xens) * scalefact)
            obs_sigma = oberrstdev * np.ones(2 * nobs, np.float64)
            if percentage_arctan <= 0:
                obs_sigma = ((obs_sigma / scalefact) / std_XXens) * scalefact
                # obs_sigma = obs_sigma /(0.001 * std_Xens * (np.cos(2. * Pvob) + 1.))
                # obs_sigma = np.where(abs(Pvob) < 1.55, obs_sigma, obs_sigma / 0.000001) / (0.01 * std_XXens)
                ## Run the EnSF filter.
                user2 = REVERSE_SDE(1000, XXens_normalization, nanals, Pvob_normalization, obs_sigma,
                                    XXens_normalization.shape[1], scalefact, indx_indxob_linear=None)
            else:
                indx_indxob_l_ensf = np.concatenate((indx_indxob_l, np.add(indx_indxob_l, nobs)), axis=0)
                indx_indxob_nl_ensf = np.setdiff1d(np.arange(2 * nobs), indx_indxob_l_ensf)
                obs_sigma[[indx_indxob_nl_ensf]] = oberrstdev_nl
                obs_sigma[[indx_indxob_l_ensf]] = ((obs_sigma[[indx_indxob_l_ensf]] / scalefact) / std_XXens[
                    [indx_indxob_l_ensf]]) * scalefact
                obs_sigma[[indx_indxob_nl_ensf]] = np.where(abs(Pvob[[indx_indxob_nl_ensf]]) < 1.55,
                                                            obs_sigma[[indx_indxob_nl_ensf]],
                                                            obs_sigma[[indx_indxob_nl_ensf]] / 0.000001) / (
                                                               0.01 * std_XXens[[indx_indxob_nl_ensf]])
                ## Run the EnSF filter.
                user2 = REVERSE_SDE(1000, XXens_normalization, nanals, Pvob_normalization, obs_sigma,
                                    XXens_normalization.shape[1], scalefact, indx_indxob_linear=indx_indxob_l_ensf)
            XXens_normalization = user2.reverse_SDE()
            XXens = XXens_normalization * std_XXens + mean_XXens

            ## Inflation: we want to restore "xens_std1" to the initial std "std_Xens_step0" in order to discentralize the ensembles.
            ## What we want to do here is to increase the std_xens without changing the mean_xens.
            mean_infla = np.mean(XXens, axis=0)
            std_infla = np.std(XXens, axis=0)
            XXens_infla = (XXens - mean_infla) / std_infla
            dynamic_inflation = std_XXens_step0 / np.std(XXens_infla, axis=0)
            xxens = XXens_infla * dynamic_inflation + mean_infla
            print("xxens.shape", xxens.shape)
            Xens = xens.reshape(nanals, 2 * nx * ny)
            print("nobs", nobs)
            for i in range(nobs):
                Xens[:, indxob[i]] = xxens[:, i]
                Xens[:, indxob[i] + N1 * N1] = xxens[:, i + nobs]
            xens = Xens.reshape(nanals, 2, nx * ny)
            print("xens.shape", xens.shape)
        elif use_letkf:
            xens = enkf_update(xens, hxens, pvob, oberrvar, covlocal_tmp, vcovlocal_fact, obcovlocal=obcovlocal)
            ## xens.shape=(20, 2, 4096), hxens.shape=(20, 2, nobs), pvob.shape=(2, nobs), oberrvar.shape=(nobs,),
            ## covlocal_tmp.shape=(nobs,4096)
            ## vcovlocal_fact = 0.0034636488, obcovlocal = None

        ## Back to 3D state vector
        for nanal in range(nanals):
            pvens[nanal] = xens[nanal].reshape((2, ny, nx))

        ## Time analysis update
        t2 = time.time()
        if profile_cpu: print('         CPU time for update: {:.1f}s'.format(t2 - t1))

        ###################################
        # Innovation statistics: analysis #
        ###################################
        print("     Innovation statistics: analysis")

        ## Analysis ensemble in observation space
        for nanal in range(nanals):
            for k in range(len(levob)):
                hxens[nanal, k, ...] = scalefact * pvens[nanal, k, ...].ravel()[indxob]  # surface pv obs
        hxensmean_a = hxens.mean(axis=0)

        ## Observation space diagnostics
        d_a = pvob - hxensmean_a
        rmsi_a = np.sqrt((d_a ** 2.0).mean())
        spread_a = (((hxens - hxensmean_a) ** 2).sum(axis=0) / (nanals - 1)).mean()  # mean analysis spread
        total_spread_a = np.sqrt(oberrvar_mean + spread_a)
        HPaHt_mean = ((hxensmean_a - hxensmean_b) * (pvob - hxensmean_a)).mean()  # mean analysis spread (D)
        HPbHt_mean = ((hxensmean_a - hxensmean_b) * (pvob - hxensmean_b)).mean()  #  mean background spread (D)
        R_mean = ((pvob - hxensmean_a) * (pvob - hxensmean_b)).mean()  # mean observation variance (D)
        # D=Desroziers method; the above 3 relations follow from linear estimation theory

        ## Model space diagnostics
        pvensmean_a = pvens.mean(axis=0)  # analysis mean
        pvprime = pvens - pvensmean_a  # analysis perturbations
        asprd = (pvprime ** 2).sum(axis=0) / (nanals - 1)  # analysis spread in model space

        #############
        # Inflation #
        #############
        print("     Inflation step")

        ## Option 1: Relaxation to prior standard deviation (Whitaker and Hamill 2012)
        if covinflate2 < 0:
            asprd = np.sqrt(asprd);
            fsprd = np.sqrt(fsprd)
            inflation_factor = 1. + covinflate1 * (fsprd - asprd) / asprd

        ## Option 2: Hodyss et al. (2016) inflation: works well under perfect model and
        ##           and linear Gaussian assumptions
        else:
            # inflation = asprd + (asprd/fsprd)**2((fsprd/nanals)+2*inc**2/(nanals-1))
            inc = pvensmean_a - pvensmean_b
            inflation_factor = covinflate1 * asprd + \
                               (asprd / fsprd) ** 2 * ((fsprd / nanals) + covinflate2 * (2. * inc ** 2 / (nanals - 1)))
            inflation_factor = np.sqrt(inflation_factor / asprd)
        # print("inflation_factor",inflation_factor)
        ## Apply inflation and reform analysis ensemble
        pvprime = pvprime * inflation_factor
        if use_letkf is True:  ##1234567890
            pvens = pvprime + pvensmean_a

        ############################################
        # More analysis diagnostics in model space #
        ############################################
        print("     More analysis diagnostics in model space")
        ## pvensmean_a.shape = pv_truth[ntime].shape = (2,64,64)
        pverr_a = (scalefact * (pvensmean_a - pv_truth[ntime])) ** 2  ##1234567890
        print("use_letkf: %s, use_ensf: %s" % (use_letkf, use_ensf))  ##1234567890
        print("obervation_network_tpye: %s" % obs_network_type)  ##1234567890
        if use_letkf is False and use_ensf is False:  ##1234567890
            print("%s %s %g" % (ntime, "RMSE_natural_run=", np.sqrt(pverr_b.mean())))  ##1234567890
        if use_letkf is False and use_ensf is True:  ##1234567890
            print("%s %s %g" % (ntime, "RMSE_Ensf=", np.sqrt(pverr_a.mean())))  ##1234567890
        if use_letkf is True and use_ensf is False:  ##1234567890
            print("%s %s %g" % (ntime, "RMSE_letkf=", np.sqrt(pverr_a.mean())))  ##1234567890
        if math.isnan(np.sqrt(pverr_a.mean())):
            print("RMSE=Nan")
            break

        # Write data of RMSE into files                                                                         ##1234567890
        if use_letkf is False and use_ensf is False:  ##1234567890  ##natural run
            rmse_file[ntime, 1] = np.sqrt(pverr_b.mean())  ##1234567890
            if ntime == nassim - 1:
                df = pd.DataFrame(rmse_file, columns=["Filtering_time_step", "RMSE_natural_run"])  ##1234567890
                # df.to_csv("RMSE_natural_run_fixed_even_1024_linear.csv", index=False)                 ##1234567890
                # df.to_csv("RMSE_natural_run_fixed_1024_linear.csv", index=False)                      ##1234567890
                # df.to_csv("RMSE_natural_run_random_1024_linear.csv", index=False)                   ##1234567890
                # df.to_csv("RMSE_natural_run_fixed_even_256_linear.csv", index=False)  ##1234567890
                # df.to_csv("RMSE_natural_run_fixed_256_linear.csv", index=False)  ##1234567890
                # df.to_csv("RMSE_natural_run_random_256_linear.csv", index=False)  ##1234567890
        if use_letkf is False and use_ensf is True:  ##1234567890  ##EnSF update
            rmse_file[ntime, 1] = np.sqrt(pverr_a.mean())  ##1234567890
            if ntime == nassim - 1:
                df = pd.DataFrame(rmse_file, columns=["Filtering_time_step", "RMSE_ensf"])  ##1234567890
                # df.to_csv("RMSE_ensf_fixed_even_1024_linear.csv", index=False)                        ##1234567890
                # df.to_csv("RMSE_ensf_fixed_1024_linear.csv", index=False)                               ##1234567890
                # df.to_csv("RMSE_ensf_random_1024_linear.csv", index=False)                          ##1234567890
                # df.to_csv("RMSE_ensf_fixed_even_256_linear.csv", index=False)  ##1234567890
                # df.to_csv("RMSE_ensf_fixed_256_linear.csv", index=False)  ##1234567890
                # df.to_csv("RMSE_ensf_random_256_linear.csv", index=False)  ##1234567890
                # df.to_csv("RMSE_ensf_fixed_1024_80linear_20arctan.csv", index=False)  ##1234567890
                # df.to_csv("RMSE_ensf_fixed_1024_60linear_40arctan.csv", index=False)  ##1234567890
                # df.to_csv("RMSE_ensf_fixed_1024_40linear_60arctan.csv", index=False)  ##1234567890
        if use_letkf is True and use_ensf is False:  ##1234567890  ##letkf update
            rmse_file[ntime, 1] = np.sqrt(pverr_a.mean())  ##1234567890
            if ntime == nassim - 1:
                df = pd.DataFrame(rmse_file, columns=["Filtering_time_step", "RMSE_letkf"])  ##1234567890
                # df.to_csv("RMSE_letkf_fixed_even_1024_linear.csv", index=False)                            ##1234567890
                # df.to_csv("RMSE_letkf_fixed_1024_linear.csv", index=False)                              ##1234567890
                # df.to_csv("RMSE_letkf_random_1024_linear.csv", index=False)                         ##1234567890
                # df.to_csv("RMSE_letkf_fixed_even_256_linear.csv", index=False)  ##1234567890
                # df.to_csv("RMSE_letkf_fixed_256_linear.csv", index=False)  ##1234567890
                # df.to_csv("RMSE_letkf_random_256_linear.csv", index=False)  ##1234567890
                # df.to_csv("RMSE_letkf_fixed_even_R1_1024_linear.csv", index=False)  ##1234567890
                # df.to_csv("RMSE_letkf_fixed_R1_1024_linear.csv", index=False)  ##1234567890
                # df.to_csv("RMSE_letkf_random_R1_1024_linear.csv", index=False)  ##1234567890
                #df.to_csv("RMSE_letkf_{}_{}_{}linear_{}arctan_{}_{}.csv".format(obs_network_type, nobs, int((1 - percentage_arctan) * 100), int(percentage_arctan * 100),
                #            int(hcovlocal_scale / 1000.), int(covinflate1 * 1000)), index=False)  ##1234567890
                df.to_csv("RMSE_letkf_12hr_{}_{}_{}linear_{}arctan_{}_{}.csv".format(obs_network_type, nobs,
                                                                                int((1 - percentage_arctan) * 100),
                                                                                int(percentage_arctan * 100),
                                                                                int(hcovlocal_scale / 1000.),
                                                                                int(covinflate1 * 1000)),index=False)  ##1234567890

        pvsprd_a = ((scalefact * (pvensmean_a - pvens)) ** 2).sum(axis=0) / (nanals - 1)
        # print("         %s %g %g %g %g %g %g %g %g %g %g %g" %\
        # (ntime,np.sqrt(pverr_a.mean()),np.sqrt(pvsprd_a.mean()),\
        # np.sqrt(pverr_b.mean()),np.sqrt(pvsprd_b.mean()),\
        # HPbHt_mean,spread_b,HPaHt_mean,spread_a,R_mean/oberrvar.mean(),bias_b,inflation_factor.mean()))

        ###################################
        # Writing data out after analysis #
        ###################################
        if savedata and (ntime >= nassim_write):
            print("     Writing data out after analysis")
            pv_a[ntime - nassim_write, :, :, :, :] = scalefact * pvens[:, :, :, :]
            tvar[ntime - nassim_write] = obtimes[ntime]
            inf[ntime - nassim_write, :, :, :] = inflation_factor[:, :, :]
            rmsi_a_nc[ntime - nassim_write] = rmsi_a
            total_spread_a_nc[ntime - nassim_write] = total_spread_a
            # nc.sync()

        ##########################
        # Ensemble forecast step #
        ##########################
        print("     *** Ensemble forecasts ***")
        t1 = time.time()
        for nanal in range(nanals):
            pvens[nanal] = models[nanal].advance(pvens[nanal])
        t2 = time.time()
        if profile_cpu: print('         CPU time for ensemble forecasts: {:.1f}s'.format(t2 - t1))

        ################################
        # KE Spectra: Error and Spread #
        ################################
        print("     Calculating KE spectra error and spread for this cycle")
        if ntime >= nassim_spinup:
            # if ntime >= 0:
            pvfcstmean = pvens.mean(axis=0)  # mean of PV forecast ensemble
            pverrspec = scalefact * rfft2(pvfcstmean - pv_truth[ntime + 1])  # PV error of forecast mean
            # in spectral space
            psispec = models[0].invert(pverrspec)
            psispec = psispec / (models[0].N * np.sqrt(2.))
            kespec = (models[0].ksqlsq * (psispec * np.conjugate(psispec))).real
            if kespec_errmean is None:
                kespec_errmean = \
                    (models[0].ksqlsq * (psispec * np.conjugate(psispec))).real
                print("kespec_errmean", kespec_errmean.shape)
            else:
                kespec_errmean = kespec_errmean + kespec
                print("kespec_errmean", kespec_errmean.shape)
            for nanal in range(nanals2):
                pvsprdspec = scalefact * rfft2(pvens[nanal] - pvfcstmean)  #  PV perturbation for a member
                # in spectral space
                psispec = models[0].invert(pvsprdspec)
                psispec = psispec / (models[0].N * np.sqrt(2.))
                kespec = (models[0].ksqlsq * (psispec * np.conjugate(psispec))).real
                if kespec_sprdmean is None:
                    kespec_sprdmean = \
                        (models[0].ksqlsq * (psispec * np.conjugate(psispec))).real / nanals2
                else:
                    kespec_sprdmean = kespec_sprdmean + kespec / nanals2
            ncount_spinup += 1

            ## Time for the cycle to complete
        tcycle_2 = time.time()
        if profile_cpu: print('     Total time for this update: {:.1f}s'.format(tcycle_2 - tcycle_1))

    print("\n All cycles are done!")

    ## Close netCDF file if we are writing the data
    if savedata:
        print(" Closing the netCDF file")
        nc.close()

    ######################################
    # Finalize after all cycles are done #
    ######################################
    print("\n === Finalize ===")

    ## Average the KE spread and error
    print("     Calculate the mean KE spectra error and mean")
    kespec_sprdmean = kespec_sprdmean / ncount_spinup
    kespec_errmean = kespec_errmean / ncount_spinup

    ## Plot the KE spectra error and spread
    if plot_KE_spectra_error_spread == True:
        print("     Plotting KE spectra error and spread")

        # Wavenumber variables
        N = int(models[0].N)  # model grid points
        k = np.abs((N * np.fft.fftfreq(N))[0:int(N / 2) + 1])  # "k" wavenumber array
        l = N * np.fft.fftfreq(N)  # "l" wavenumber array
        k, l = np.meshgrid(k, l)  #  meshgrid of wavenumbers
        ktot = np.sqrt(k ** 2 + l ** 2)  # total wavenumber
        ktotmax = int((N / 2) + 1)  # maximum wavenumber

        # Initialize KE spectra error and spread arrays
        kespec_err = np.zeros(ktotmax, np.float64)
        kespec_sprd = np.zeros(ktotmax, np.float64)

        # Fill in those arrays
        for i in range(kespec_errmean.shape[2]):
            for j in range(kespec_errmean.shape[1]):
                totwavenum = ktot[j, i]
                if int(totwavenum) < ktotmax:
                    kespec_err[int(totwavenum)] = kespec_err[int(totwavenum)] + \
                                                  kespec_errmean[:, j, i].mean(axis=0)
                    kespec_sprd[int(totwavenum)] = kespec_sprd[int(totwavenum)] + \
                                                   kespec_sprdmean[:, j, i].mean(axis=0)
        # print('         Mean KE error error={:.1f} and spread={:1f}'.\
        #                format(kespec_errmean.sum(),kespec_sprdmean.sum()))

        # Write data of KE_spectra into files
        KEspectra_file = np.zeros((ktotmax, 2), np.float64)
        wavenums = np.arange(ktotmax, dtype=np.float64)
        KEspectra_file[:, 0] = wavenums
        KEspectra_file[:, 1] = kespec_err  ##1234567890
        """
        if use_letkf is False and use_ensf is True:  ##1234567890  ##EnSF update
            df = pd.DataFrame(KEspectra_file, columns=["Wavenumber", "KEspectra_ensf"])                     ##1234567890
            #df.to_csv("KEspectra_ensf_fixed_even_1024_linear.csv", index=False)                        ##1234567890
            #df.to_csv("KEspectra_ensf_fixed_1024_linear.csv", index=False)                               ##1234567890
            # df.to_csv("KEspectra_ensf_random_1024_linear.csv", index=False)                          ##1234567890
            # df.to_csv("KEspectra_ensf_fixed_even_256_linear.csv", index=False)                            ##1234567890
            # df.to_csv("KEspectra_ensf_fixed_256_linear.csv", index=False)                                 ##1234567890
            # df.to_csv("KEspectra_ensf_random_256_linear.csv", index=False)                                    ##1234567890
            #df.to_csv("KEspectra_ensf_random_64_linear.csv", index=False)                                  ##1234567890
            #df.to_csv("KEspectra_ensf_random_16_linear.csv", index=False)                                          ##1234567890
            #df.to_csv("KEspectra_ensf_random_32_linear.csv", index=False)                                          ##1234567890
            #df.to_csv("KEspectra_ensf_random_128_linear.csv", index=False)                                         ##1234567890
            #df.to_csv("KEspectra_ensf_random_512_linear.csv", index=False)                                     ##1234567890
            #df.to_csv("KEspectra_ensf_random_2048_linear.csv", index=False)                                        ##1234567890
            #df.to_csv("KEspectra_ensf_random_4096_linear.csv", index=False)                             ##1234567890
            # df.to_csv("KEspectra_ensf_fixed_16_linear.csv", index=False)
            # df.to_csv("KEspectra_ensf_fixed_32_linear.csv", index=False)
            # df.to_csv("KEspectra_ensf_fixed_64_linear.csv", index=False)
            # df.to_csv("KEspectra_ensf_fixed_128_linear.csv", index=False)
            # df.to_csv("KEspectra_ensf_fixed_256_linear.csv", index=False)
            # df.to_csv("KEspectra_ensf_fixed_512_linear.csv", index=False)
            # df.to_csv("KEspectra_ensf_fixed_1024_linear.csv", index=False)
            # df.to_csv("KEspectra_ensf_fixed_2048_linear.csv", index=False)
            #df.to_csv("KEspectra_ensf_fixed_4096_linear.csv", index=False)
        if use_letkf is True and use_ensf is False:  ##1234567890  ##letkf update
            df = pd.DataFrame(KEspectra_file, columns=["Wavenumber", "KEspectra_letkf"])                            ##1234567890
            #df.to_csv("KEspectra_letkf_fixed_even_1024_linear.csv", index=False)                            ##1234567890
            #df.to_csv("KEspectra_letkf_fixed_1024_linear.csv", index=False)                              ##1234567890
            #df.to_csv("KEspectra_letkf_random_1024_linear.csv", index=False)                         ##1234567890
            # df.to_csv("KEspectra_letkf_fixed_even_256_linear.csv", index=False)                       ##1234567890
            # df.to_csv("KEspectra_letkf_fixed_256_linear.csv", index=False)                                ##1234567890
            #df.to_csv("KEspectra_letkf_random_4096_linear.csv", index=False)                                           ##1234567890
            # df.to_csv("KEspectra_letkf_random_256_linear.csv", index=False)                                   ##1234567890
            #df.to_csv("KEspectra_letkf_random_64_linear.csv", index=False)                         ##1234567890
            #df.to_csv("KEspectra_letkf_random_16_linear.csv", index=False)                                     ##1234567890
            #df.to_csv("KEspectra_letkf_random_32_linear.csv", index=False)                          ##1234567890
            #df.to_csv("KEspectra_letkf_random_128_linear.csv", index=False)  ##1234567890
            #df.to_csv("KEspectra_letkf_random_512_linear.csv", index=False)  ##1234567890
            #df.to_csv("KEspectra_letkf_random_2048_linear.csv", index=False)  ##1234567890
            # df.to_csv("RMSE_letkf_fixed_even_R1_1024_linear.csv", index=False)                                    ##1234567890
            # df.to_csv("RMSE_letkf_fixed_R1_1024_linear.csv", index=False)                                     ##1234567890
            # df.to_csv("RMSE_letkf_random_R1_1024_linear.csv", index=False)                                                 ##1234567890
            #df.to_csv("KEspectra_letkf_fixed_16_linear.csv", index=False)
            #df.to_csv("KEspectra_letkf_fixed_32_linear.csv", index=False)
            #df.to_csv("KEspectra_letkf_fixed_64_linear.csv", index=False)
            #df.to_csv("KEspectra_letkf_fixed_128_linear.csv", index=False)
            #df.to_csv("KEspectra_letkf_fixed_256_linear.csv", index=False)
            #df.to_csv("KEspectra_letkf_fixed_512_linear.csv", index=False)
            #df.to_csv("KEspectra_letkf_fixed_1024_linear.csv", index=False)
            #df.to_csv("KEspectra_letkf_fixed_2048_linear.csv", index=False)
            #df.to_csv("KEspectra_letkf_fixed_4096_linear.csv", index=False)
        """

        # for n in range(1,ktotmax):
        #    print('# ',wavenums[n],kespec_err[n],kespec_sprd[n])
        # plt.loglog(wavenums[1:-1], kespec_err[1:-1], color='r')
        # plt.loglog(wavenums[1:-1], kespec_sprd[1:-1], color='b')
        # plt.title('error (red) and spread (blue) spectra')
        # if infl_method == 'rtps':
        #    plt.savefig('{}/errorspread_spectra_p{}.png'.format(outdir_figs, nobs))
        # elif infl_method == 'hodyss':
        #    plt.savefig('{}/errorspread_spectra_p{}.png'.format(outdir_figs, nobs))
        # plt.show()
        # plt.close()


################
# Execute code #
################
if __name__ == "__main__":
    main()