# -*- coding: utf-8 -*-
"""
Created on Sat Apr  1 18:52:11 2023

@author: KatieCampos
"""

# Title: Stellar Structure Calculation
# Purpose: Final Project, Stellar Structure and Evolution
# Summary: This script will calculate the ZAMS structure of an approximately 
# solar mass star and produce two figures to demonstrate this.
# Author: Kathleen Hamilton-Campos
# Dates: April-May 2023
# Guide to Abbreviations in comments:
    # SI = Stellar Interiors
    # SSAE = Stellar Structure and Evolution
    # NR = Numerical Recipes

# import necessary libraries
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.interpolate import griddata
from scipy import optimize

# make plots look pretty
plt.rcParams['text.usetex'] = True


# set arguments
def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                    description='set parameters for galactic analysis')
        
    parser.add_argument('--projdir', metavar='projdir', type=str, action='store',
                        help='Where is our working directory?')
    parser.set_defaults(projdir='/Users/Kitty/Documents/College/Hopkins/Spring2023/AS.171.611.01/Project')
    
    parser.add_argument('--bigfont', metavar='bigfont', type=int, action='store',
                        help='How big should the fontsize be?')
    parser.set_defaults(bigfont=20)
    
    args = parser.parse_args()
    return args


# set values
args = parse_args()
projdir = args.projdir
bigfont = args.bigfont


# import Professor Schlaufman's constants file and extra(s) from astropy
exec(open('{}/constants.py'.format(projdir)).read())
from astropy.constants import N_A
N_A = N_A.value


# calculate density using Eq. 3.104, SI
# assumptions: simple mixture of ideal gas and radiation, fully ionized
def density(P, T):
    X = 0.70
    mu = 4 / (3 + 5*X)
    dens = (P - (a/3) * T**4) * mu / (N_A * k * T)
    return dens


# interpolate opacity given density and temperature
def opacity(rho, T):
    # R is defined as rho / T_6^3
    T6 = T/1e6
    logR = np.log10(rho / T6**3)
    logT = np.log10(T)
    kappa = 10**griddata(points, values, (logR, logT), method = 'linear')
    return kappa


# calculate the gradient of temperature wrt to pressure from Eq. 4.30 in SI
def nabla(M, L, P, rho, T):
    kappa = opacity(rho, T)
    del_rad = (3 * P * kappa * L) / (16 * np.pi * a * c * T**4 * G * M)
    return del_rad


# calculate f11 using weak screening from Section 18.4 in SSAE
def screen_fac(rho, T):
    # Z1 = Z2 = 1 for pp-chain -> Z1*Z2 = 1; zeta is order 1
    Z1 = 1
    Z2 = 1
    zeta = 1
    prefac = 5.92e-3
    expo = prefac*Z1*Z2*np.sqrt(zeta*rho*(T/1e7)**-3)
    f11 = np.exp(expo)
    return f11


# calculate gaunt factor for pp-chain from Eq. 18.63 in SSAE
def g_fac_pp(T):
    T9 = T/1e9
    g11 = 1 + 3.82*T9 + 1.51*T9**2 + 0.144*T9**3 - 0.0114*T9**4
    return g11


# calculate energy generation rate for pp-chain from Eq. 18.63 in SSAE
def pp_gen(rho, T):
    X = 0.70
    # psi is 1 for pp1
    psi = 1
    f11 = screen_fac(rho, T)
    g11 = g_fac_pp(T)
    T9 = T/1e9
    vari = psi * f11 * g11 * rho * X**2 * (T9)**(-2/3)
    ep_pp = 2.57e4 * vari * np.exp(-3.381/((T9)**(1/3)))
    return ep_pp


# calculate gaunt factor for CNO cycle from Eq. 18.65 in SSAE
def g_fac_cno(T):
    T9 = T/1e9
    g141 = 1 - 2.00*T9 + 3.41*T9**2 - 2.43*T9**3
    return g141


# calculate energy generation rate for CNO cycle from Eq. 18.65 in SSAE
def cno_gen(T, rho):
    X = 0.70
    Z = 0.02
    g141 = g_fac_cno(T)
    T9 = T/1e9
    # X_cno = X_C + X_N + X_O -> between two-thirds and three-fourths of the metals
    X_cno = 0.7*Z
    expo_cno = -15.231*T9**(-1/3) - (T9/0.8)**2
    ep_cno = 8.24e25 * g141 * X_cno * X * rho * T9**(-2/3) * np.exp(expo_cno)
    return ep_cno


# calculate total rate of energy generation
def en_gen(rho, T):
    pp = pp_gen(rho, T)
    cno = cno_gen(T,rho)
    tot_gen = pp + cno
    return tot_gen


# determine values for luminosity, pressure, radius, and temperature for a 
# starting point just outside of the star's center to avoid mathematical issues
def inners(Tc, Pc):
    rho_c = density(Pc, Tc)
    in_point = 1e-10
    m_star = in_point * Ms
    epsilon = en_gen(rho_c, Tc)
    l_star = epsilon * m_star

    # calculate the starting point of the radius from Eq. 11.3 in SSAE
    r_star = (3 * m_star / (4*np.pi*rho_c))**(1/3)

    # calculate the pressure from Eq. 11.6 in SSAE
    P = Pc - ((3 * G / (8*np.pi)) * ((4 * np.pi * rho_c / 3)**(4/3)) * 
              (m_star**(2/3)))

    # calculate temperature gradient
    del_rad = nabla(m_star, l_star, P, rho_c, Tc)

    # check whether the core is radiative or convective
    if del_rad > del_ad:
        # the core is convective - temperature from Eq. 11.9b in SSAE
        Tfac = (np.pi / 6)**(1/3) * (G * del_ad / Pc) * rho_c**(4/3) * m_star**(2/3)
        lnT = np.log(Tc) - Tfac
        T = np.exp(lnT)
    else:
        # the core is radiative - temperature from Eq. 11.9a in SSAE
        kappa_c = opacity(rho_c, Tc)
        Tfac = (kappa_c * epsilon / (2 * a * c)) * (3 / (4 * np.pi))**(2/3)
        Tvars = rho_c**(4/3) * m_star**(2/3)
        T = (Tc**4 - (Tfac * Tvars))**(1/4)

    # return values for luminousity, pressure, radius, and temperature at the
    # stellar interior
    return np.array([l_star, P, r_star, T])


# calculate which density is the lowest: that due to opacity or based on
# the equation of state
def low_rho(rho, T, L_star, M_star, g, mu):
    kappa = opacity(rho, T)
    rho_fac = 1 + (kappa * L_star / (4 * np.pi *c * G * M_star))
    rho_opacity = ((2 * g) / (3 * kappa)) * rho_fac
    rho_eos = (a * T**4 / 3) + (rho * N_A * k * T / mu)
    diff = (1 - (rho_opacity / rho_eos))**2
    return diff

# determine values for luminosity, pressure, radius, and temperature for a 
# starting point just inside of the star's photosphere to avoid mathematical 
# issues
def outers(M_star, L_star, R_star):
    X = 0.70
    mu = 4 / (3 + 5 * X)
    g = G * M_star / R_star**2
    Teff = (L_star / (4 * np.pi * sb * R_star**2))**(1/4)
    rho_args = (Teff, L_star, M_star, g, mu)
    rho_min = optimize.minimize(low_rho, 1e-8, args = rho_args, 
                                method = 'Nelder-Mead', bounds=[(1e-13,1e-5)])
    rho = rho_min.x[0]
    kappa = opacity(rho, Teff)
    Pfac = 1 + (kappa * L_star / (4 * np.pi * c * G * M_star))
    P = (2 * g / (3 * kappa)) * Pfac
    
    # return values for luminousity, pressure, radius, and temperature at the
    # stellar interior
    return np.array([L_star, P, R_star, Teff])


# Lagrangian forms of the four coupled ordinary differential equations of 
# stellar structure
def diff_eq(m_star, inits):
    l_star, P, r_star, T = inits
    rho = density(P, T)
    del_rad = nabla(m_star, l_star, P, rho, T)
    del_rad_val = np.minimum(del_rad, del_ad)

    # Eq. 7.5, 7.6, 7.7, and 7.11 from SI
    dLrdMr = en_gen(rho, T)
    dPdMr = -(G * m_star) / (4 * np.pi * r_star**4)
    drdMr = (4*np.pi*r_star**2*rho)**-1
    dTdMr = del_rad_val * dPdMr * (T/P)
    
    # return derivatives for luminosity, pressure, radius, and temperature
    # wrt to mass
    return np.array([dLrdMr, dPdMr, drdMr, dTdMr])


# creates initial values for stellar parameters and integrates in both
# directions as in Ch. 18 of NR
def shootf(inits, M_star, M_meet, runs, in_point, out_point):
    L_star, Pc, R_star, Tc = inits

    inner_inits = inners(Tc, Pc)
    outer_inits = outers(M_star, L_star, R_star)

    # solve across mass arrays
    inside_out = np.linspace(in_point * Ms,  Ms * M_meet, num=int(runs))
    outside_in = np.linspace(Ms, Ms * M_meet, num=int(runs))
    
    solve_inner = solve_ivp(diff_eq, (inside_out[0], inside_out[-1]), 
                            inner_inits, method='RK45', t_eval=inside_out)
    solve_outer = solve_ivp(diff_eq, (outside_in[0], outside_in[-1]), 
                            outer_inits, method='RK45', t_eval=outside_in)
    
    inside_out_sol = solve_inner.y
    outside_in_sol = solve_outer.y
    
    return inside_out, outside_in, inside_out_sol, outside_in_sol


# shoot for a solution from both inside and outside of the star, then
# calculate the residuals to check how close of a match both are
def residuals(inits, M_star, M_meet, runs, in_point, out_point):
    inside_out, outside_in, inside_out_sol, outside_in_sol = shootf(inits, 
                                                                    M_star, 
                                                                    M_meet, 
                                                                    runs, 
                                                                    in_point, 
                                                                    out_point)
    
    # check how close the interior solution matches the exterior one
    dL = (inside_out_sol[0,-1] - outside_in_sol[0,-1]) / L_star
    dP = (inside_out_sol[1,-1] - outside_in_sol[1,-1]) / Pc
    dR = (inside_out_sol[2,-1] - outside_in_sol[2,-1]) / R_star
    dT = (inside_out_sol[3,-1] - outside_in_sol[3,-1]) / Tc

    # return the residuals
    return np.array([dL, dP, dR, dT])


# creates the stellar solution once the differences between the inward and 
# outward integrations have been sufficiently minimized
def solvef(inits, M_star, M_meet, runs, in_point, out_point):
    inside_out, outside_in, inside_out_sol, outside_in_sol = shootf(inits, 
                                                                    M_star, 
                                                                    M_meet, 
                                                                    runs, 
                                                                    in_point, 
                                                                    out_point)

    # combine mass arrays
    mass = np.concatenate([inside_out, np.flipud(outside_in)], axis=0)

    # save mass
    solution = np.zeros((9, mass.shape[0]))
    solution[0] = mass

    # save luminosity, pressure, radius, and temperature
    sols = np.concatenate([inside_out_sol, np.fliplr(outside_in_sol)], axis=1)
    solution[1:5] = sols

    # save density
    rho = density(solution[2],solution[4])
    solution[5] = rho

    # save temperature-pressure gradient
    del_rad = nabla(mass, solution[1], solution[2], rho, solution[4])
    solution[6] = del_rad
    
    # save energy
    ep = en_gen(solution[5], solution[4])
    solution[7] = ep
    
    # save opacity
    kap = opacity(solution[5], solution[4])
    solution[8] = kap

    return solution


# start the program
if __name__ == '__main__':
    # set stellar parameters using homology relations
    M_star_solar = 1.4
    M_star = M_star_solar*Ms
    L_star = M_star_solar**(3.5)*Ls
    R_star = M_star_solar**(0.75)*Rs
    
    # assume a constant density sphere
    Pc = (3 / (8 * np.pi)) * (G * (M_star)**2) / (R_star)**4
    Tc = (1/2) * ((4 / (3 + 5 * 0.7)) / (N_A * k)) * (G * M_star) / (R_star)
    
    # set baseline for ideal gas, assume complete ionization
    del_ad = 0.4
    
    # set metallicity fractions: Table 73 in OPAL
    X = 0.70
    Y = 0.28
    Z = 1 - X - Y
    
    # read in opacities from Table 73
    OPAL = pd.read_csv('{}/Table73OPAL.txt'.format(projdir), index_col=0)
    OPAL.columns = OPAL.columns.astype(float)

    # create a grid of opacities for interpolation
    RT = []
    opacities = []
    for i in range(len(OPAL.columns.values)):
        for j in range(len(OPAL.index.values)):
            RT.append([OPAL.columns.values[i], OPAL.index.values[j]])
            opacities.append(OPAL.values[j][i])

    # create an array of the initial guesses for the stellar parameters
    inits = np.array([L_star, Pc, R_star, Tc])
    
    # choose where to have the inside integration and outside integration meet
    M_meet = 0.33
    
    # number of runs
    runs = 1e5
    
    # offset from center to avoid numerical issues
    in_point = 1e-10
    
    # offset from surface to avoid numerical issues
    out_point = 0.99
    
    # arguments needed for the numerical solution
    num_args = (M_star, M_meet, runs, in_point, out_point)

    # boundaries for minimizing residuals
    low_bounds = [L_star * 0.1, Pc, R_star * 0.1, Tc]
    hi_bounds = [L_star, Pc * 10000, R_star, Tc * 1000]
    bounds = (low_bounds, hi_bounds)
    
    args = (M_star, M_meet, int(runs), in_point, out_point)
    bounds = (low_bounds, hi_bounds)
    
    # allows the figure to be made with saved data instead of running the 
    # calculations again
    if False:
        # method of minimization will be least squares - similar Newton's Method
        fins = optimize.least_squares(residuals, inits, args = args, 
                                      bounds = bounds)

        # once the interior and exterior integrations agree sufficiently well,
        # create the solution
        stellar = solvef(fins.x, args[0], args[1], args[2], args[3], args[4])

        # save results to a machine-readable numpy file
        with open('{}/StellarStructure{}.npy'.format(projdir, M_star_solar), 
                  'wb') as rec:
            np.save(rec, stellar)
    
    # load in table of stellar results
    star = np.load('{}/StellarStructure1.4.npy'.format(projdir), 
                   allow_pickle = True)[()]
    converged = [1.2422044782076352e+34, 1.915468190754002e+17, 
                 100348408236.7951, 17276515.770172738, ]
    MESA = [1.44019E+34, 1.97488E+17, 9.74700E+10, 1.75278E+07]
    inside_out, outside_in, inside_out_sol, outside_in_sol = shootf(converged, 
                                                                    M_star, 
                                                                    M_meet, 
                                                                    runs, 
                                                                    in_point, 
                                                                    out_point)

    # re-assign parameters
    mass = star[0]
    luminosity = star[1]
    pressure = star[2]
    radius = star[3]
    temperature = star[4]
    rho = star[5]
    del_rad = star[6]
    epsilon = star[7]
    kappa = star[8]
    
    # normalize parameters
    norm_m = mass/np.max(mass)
    norm_lum = luminosity/np.max(luminosity)
    norm_p = pressure/np.max(pressure)
    norm_r = radius/np.max(radius)
    norm_t = temperature/np.max(temperature)
    norm_rho = rho/np.max(rho)
    norm_del = del_rad/np.max(del_rad)
    norm_ep = epsilon/np.max(epsilon)
    norm_kap = kappa/np.max(kappa)
    
    # plot results as a function of mass
    figm, axm = plt.subplots(1, 1, figsize = (10, 10))
    figm.set_dpi(300)
    plt.plot(norm_m, norm_lum, linestyle = 'solid', label = r'Luminosity $[erg s^{-1}]$')
    plt.plot(norm_m, norm_p, linestyle = 'dotted', label = 'Pressure $[Ba]$')
    plt.plot(norm_m, norm_r, linestyle = 'dashed', label = 'Radius $[cm^{-3}]$')
    plt.plot(norm_m, norm_t, linestyle = 'dashdot', label = 'Temperature $[K]$')
    plt.plot(norm_m, norm_rho, linestyle = 'solid', label = r'Rho $[g cm^{-3}]$')
    plt.plot(norm_m, norm_del, linestyle = 'dotted', label = r'$\Delta_{rad}$')
    plt.plot(norm_m, norm_ep, linestyle = 'dashed', label = r'$\varepsilon$')
    plt.plot(norm_m, norm_kap, linestyle = 'dashdot', label = r'$\kappa$')
    plt.title(r'Stellar Parameters as a Function of Mass for a 1.4 $M_{\odot}$ Star', 
              fontsize = bigfont)
    axm.set_xlabel(r'Normalized Mass $(M/M_\odot)$', fontsize = bigfont)
    axm.set_ylabel('Normalized Quantities', fontsize = bigfont)
    plt.setp(axm.get_xticklabels(), fontsize=bigfont)
    plt.setp(axm.get_yticklabels(), fontsize=bigfont)
    plt.legend(bbox_to_anchor=(0.55, 0.5), fontsize = 15)
    plt.savefig('{}/StellarMass1.4.png'.format(projdir))
    plt.show()

    # plot inside out and outside in results
    fig4, ((axl, axp), (axr, axt)) = plt.subplots(2, 2, figsize = (15, 10))
    fig4.set_dpi(300)
    axl.plot(inside_out, inside_out_sol[0])
    axl.plot(outside_in, outside_in_sol[0])
    axl.set_xlabel('Mass (g)', fontsize = bigfont)
    axl.set_ylabel(r'Luminosity (erg $s^{-1}$)', fontsize = bigfont)
    plt.setp(axl.get_xticklabels(), fontsize=bigfont)
    plt.setp(axl.get_yticklabels(), fontsize=bigfont)
    axp.plot(inside_out, inside_out_sol[1])
    axp.plot(outside_in, outside_in_sol[1])
    axp.set_xlabel('Mass (g)', fontsize = bigfont)
    axp.set_ylabel(r'Pressure (dyne $cm^{-2}$)', fontsize = bigfont)
    plt.setp(axp.get_xticklabels(), fontsize=bigfont)
    plt.setp(axp.get_yticklabels(), fontsize=bigfont)
    axr.plot(inside_out, inside_out_sol[2])
    axr.plot(outside_in, outside_in_sol[2])
    axr.set_xlabel('Mass (g)', fontsize = bigfont)
    axr.set_ylabel('Radius (cm)', fontsize = bigfont)
    plt.setp(axr.get_xticklabels(), fontsize=bigfont)
    plt.setp(axr.get_yticklabels(), fontsize=bigfont)
    axt.plot(inside_out, inside_out_sol[3])
    axt.plot(outside_in, outside_in_sol[3])
    axt.set_xlabel('Mass (g)', fontsize = bigfont)
    axt.set_ylabel('Temperature (K)', fontsize = bigfont)
    plt.setp(axt.get_xticklabels(), fontsize=bigfont)
    plt.setp(axt.get_yticklabels(), fontsize=bigfont)
    plt.savefig('{}/Integrations1.4.png'.format(projdir))
    plt.show()