"""Block L4 — transport properties (viscosity, thermal conductivity, surface tension) of compounds other than water in
`compound_properties` (packages.properties.pure): validation against NIST and against the reference correlations.

Sources:
- NIST Chemistry WebBook, SRD 69, «Thermophysical Properties of Fluid Systems» (E. W. Lemmon, M. O. McLinden, D. G.
  Friend; REFPROP-based), read on 2026-11-15: saturation tables (Type=SatT) and isobars at 0.001 MPa and 5-10 MPa
  (Type=IsoBar), https://webbook.nist.gov/cgi/fluid.cgi?ID=<id>&Action=Data&Type=... (one URL per fluid in NIST_ROWS).
  The rows below were copied from the downloaded tab-separated tables (saturated-liquid rows at the 5/40/80/95 % points
  of the table, gas rows at the first/middle/last temperature); none is from memory. Ethanol and acetone are not
  in the WebBook fluid list.
- CoolProp 8.0.0 (installed): the reference transport correlation of each fluid (BibTeX key in its fluid file) and
  its surface-tension correlation (Mulero et al. 2012/2014).
Measured (2026-11-15) and asserted below: the tool against NIST by class; CoolProp's correlations against NIST (several
are older than REFPROP's and deviate: n-pentane viscosity −45 %, ammonia ±13 %, H2S viscosity 9 %, methane/ethane/
n-butane/n-heptane viscosity 3-5 %), hence the rule of the tool: use CoolProp's correlation only where it matched NIST
within 1 %, otherwise give thermo's value with CoolProp as reference_check and flag disagreements.
"""
from __future__ import annotations

import json
import math

import pytest

from packages.agent import wrappers_properties as WP
from packages.dwsim_client import tools_properties as TPR
from packages.properties import pure as PU

CP = pytest.importorskip("CoolProp.CoolProp")

# (compound, kind, T_K, P_Pa, viscosity_Pa_s, thermal_conductivity_W_mK, surface_tension_N_m) from NIST WebBook:
# kind 'sat' = saturated liquid, 'lowP' = gas at 1 kPa (dilute), 'highP' = gas or supercritical fluid at 5-10 MPa.
NIST_ROWS = [
    # nitrogen: https://webbook.nist.gov/cgi/fluid.cgi?ID=C7727379&Action=Data (Type=SatT; IsoBar P=0.001 MPa and 5-10 MPa)
    ("nitrogen", "sat", 82.675, 181680, 0.00013159, 0.13428, 0.007691),
    ("nitrogen", "sat", 108.79, 1.3658e+06, 5.816e-05, 0.08281, 0.0024541),
    ("nitrogen", "sat", 121.6, 2.7191e+06, 3.5305e-05, 0.05816, 0.00046626),
    ("nitrogen", "sat", 125.13, 3.2266e+06, 2.6493e-05, 0.056891, 7.5469e-05),
    ("nitrogen", "lowP", 100, 1000, 6.904e-06, 0.0092784, None),
    ("nitrogen", "lowP", 400, 1000, 2.22e-05, 0.032782, None),
    ("nitrogen", "lowP", 700, 1000, 3.2828e-05, 0.050292, None),
    ("nitrogen", "highP", 150, 1e+07, 2.774e-05, 0.047718, None),
    ("nitrogen", "highP", 600, 1e+07, 3.0284e-05, 0.046995, None),
    # oxygen: https://webbook.nist.gov/cgi/fluid.cgi?ID=C7782447&Action=Data (Type=SatT; IsoBar P=0.001 MPa and 5-10 MPa)
    ("oxygen", "sat", 99.919, 252290, 0.00015285, 0.13668, 0.010755),
    ("oxygen", "sat", 132.92, 2.0173e+06, 7.2314e-05, 0.088402, 0.0034602),
    ("oxygen", "sat", 148.87, 4.0344e+06, 4.5056e-05, 0.065638, 0.00067536),
    ("oxygen", "sat", 153.26, 4.7909e+06, 3.5093e-05, 0.062305, 0.00011279),
    ("oxygen", "lowP", 100, 1000, 7.7025e-06, 0.0089447, None),
    ("oxygen", "lowP", 400, 1000, 2.5821e-05, 0.033995, None),
    ("oxygen", "lowP", 700, 1000, 3.8641e-05, 0.053947, None),
    ("oxygen", "highP", 160, 1e+07, 4.7957e-05, 0.066518, None),
    ("oxygen", "highP", 660, 1e+07, 3.8393e-05, 0.053467, None),
    # argon: https://webbook.nist.gov/cgi/fluid.cgi?ID=C7440371&Action=Data (Type=SatT; IsoBar P=0.001 MPa and 5-10 MPa)
    ("argon", "sat", 99.399, 308600, 0.00018426, 0.11109, 0.0096189),
    ("argon", "sat", 129.59, 1.9865e+06, 8.6783e-05, 0.070938, 0.003168),
    ("argon", "sat", 145.09, 3.9042e+06, 5.1832e-05, 0.051908, 0.0006029),
    ("argon", "sat", 149.39, 4.6233e+06, 3.9566e-05, 0.052454, 9.7212e-05),
    ("argon", "lowP", 100, 1000, 8.1898e-06, 0.0063667, None),
    ("argon", "lowP", 400, 1000, 2.8692e-05, 0.022491, None),
    ("argon", "lowP", 700, 1000, 4.355e-05, 0.034118, None),
    ("argon", "highP", 160, 1e+07, 5.0841e-05, 0.049639, None),
    ("argon", "highP", 660, 1e+07, 4.2621e-05, 0.034465, None),
    # co2: https://webbook.nist.gov/cgi/fluid.cgi?ID=C124389&Action=Data (Type=SatT; IsoBar P=0.001 MPa and 5-10 MPa)
    ("carbon dioxide", "sat", 229.04, 860920, 0.00020605, 0.16174, 0.013608),
    ("carbon dioxide", "sat", 270.67, 3.2617e+06, 0.00010478, 0.11209, 0.0049384),
    ("carbon dioxide", "sat", 295.17, 6.0054e+06, 6.3139e-05, 0.083107, 0.00094649),
    ("carbon dioxide", "sat", 302.05, 7.0343e+06, 4.7603e-05, 0.085329, 0.00015171),
    ("carbon dioxide", "lowP", 250, 1000, 1.2581e-05, 0.012989, None),
    ("carbon dioxide", "lowP", 550, 1000, 2.5927e-05, 0.036902, None),
    ("carbon dioxide", "lowP", 750, 1000, 3.3257e-05, 0.052639, None),
    ("carbon dioxide", "highP", 310, 1e+07, 5.4417e-05, 0.075453, None),
    ("carbon dioxide", "highP", 710, 1e+07, 3.2988e-05, 0.052333, None),
    # hydrogen: https://webbook.nist.gov/cgi/fluid.cgi?ID=C1333740&Action=Data (Type=SatT; IsoBar P=0.001 MPa and 5-10 MPa)
    ("hydrogen", "sat", 19.257, 71810, 1.4819e-05, 0.10332, 0.0021042),
    ("hydrogen", "sat", 27.487, 522970, 7.8612e-06, 0.099538, 0.00072261),
    ("hydrogen", "sat", 31.643, 1.0386e+06, 5.3128e-06, 0.090315, 0.00013664),
    ("hydrogen", "sat", 32.796, 1.2319e+06, 4.2141e-06, 0.095362, 4.3742e-05),
    ("hydrogen", "lowP", 20, 1000, 1.0016e-06, 0.015539, None),
    ("hydrogen", "lowP", 320, 1000, 9.3462e-06, 0.19599, None),
    ("hydrogen", "lowP", 520, 1000, 1.3091e-05, 0.27864, None),
    ("hydrogen", "highP", 50, 1e+07, 6.8973e-06, 0.10786, None),
    ("hydrogen", "highP", 500, 1e+07, 1.2761e-05, 0.27786, None),
    # water: https://webbook.nist.gov/cgi/fluid.cgi?ID=C7732185&Action=Data (Type=SatT; IsoBar P=0.001 MPa and 5-10 MPa)
    ("water", "sat", 457.36, 1.1038e+06, 0.00014675, 0.6692, 0.041248),
    ("water", "sat", 575.1, 8.826e+06, 8.5122e-05, 0.54967, 0.013914),
    ("water", "sat", 628.53, 1.7651e+07, 6.3026e-05, 0.44862, 0.0026769),
    ("water", "sat", 642.82, 2.0961e+07, 5.2625e-05, 0.44252, 0.00042918),
    ("water", "lowP", 300, 1000, 9.7659e-06, 0.018563, None),
    ("water", "lowP", 600, 1000, 2.1433e-05, 0.046278, None),
    ("water", "lowP", 900, 1000, 3.3677e-05, 0.082578, None),
    ("water", "highP", 600, 1e+07, 2.1017e-05, 0.072247, None),
    ("water", "highP", 900, 1e+07, 3.4201e-05, 0.09042, None),
    # methane: https://webbook.nist.gov/cgi/fluid.cgi?ID=C74828&Action=Data (Type=SatT; IsoBar P=0.001 MPa and 5-10 MPa)
    ("methane", "sat", 123.35, 241070, 9.2402e-05, 0.16718, 0.010663),
    ("methane", "sat", 163.78, 1.8467e+06, 4.3356e-05, 0.10951, 0.0034853),
    ("methane", "sat", 183.51, 3.6817e+06, 2.8086e-05, 0.080288, 0.00066221),
    ("methane", "sat", 188.93, 4.3698e+06, 2.2409e-05, 0.078511, 9.4627e-05),
    ("methane", "lowP", 150, 1000, 5.8149e-06, 0.015621, None),
    ("methane", "lowP", 350, 1000, 1.2688e-05, 0.041523, None),
    ("methane", "lowP", 550, 1000, 1.8167e-05, 0.078271, None),
    ("methane", "highP", 200, 1e+07, 2.9554e-05, 0.083929, None),
    ("methane", "highP", 600, 1e+07, 2.0307e-05, 0.09209, None),
    # ethane: https://webbook.nist.gov/cgi/fluid.cgi?ID=C74840&Action=Data (Type=SatT; IsoBar P=0.001 MPa and 5-10 MPa)
    ("ethane", "sat", 202.57, 243610, 0.00013471, 0.15006, 0.013363),
    ("ethane", "sat", 264.99, 1.9489e+06, 6.5031e-05, 0.097047, 0.004275),
    ("ethane", "sat", 294.75, 3.8978e+06, 4.0159e-05, 0.073795, 0.00079313),
    ("ethane", "sat", 302.88, 4.6286e+06, 3.073e-05, 0.074034, 0.00012092),
    ("ethane", "lowP", 200, 1000, 6.2654e-06, 0.010361, None),
    ("ethane", "lowP", 400, 1000, 1.2143e-05, 0.035771, None),
    ("ethane", "lowP", 600, 1000, 1.7246e-05, 0.07386, None),
    ("ethane", "highP", 310, 1e+07, 4.6171e-05, 0.079795, None),
    ("ethane", "highP", 610, 1e+07, 1.9731e-05, 0.07929, None),
    # propane: https://webbook.nist.gov/cgi/fluid.cgi?ID=C74986&Action=Data (Type=SatT; IsoBar P=0.001 MPa and 5-10 MPa)
    ("propane", "sat", 249.32, 212560, 0.00016099, 0.11874, 0.013234),
    ("propane", "sat", 322.8, 1.7005e+06, 7.3941e-05, 0.082749, 0.0041811),
    ("propane", "sat", 357.59, 3.401e+06, 4.3996e-05, 0.068562, 0.000797),
    ("propane", "sat", 367.05, 4.0386e+06, 3.2937e-05, 0.069992, 0.00013041),
    ("propane", "lowP", 250, 1000, 6.8338e-06, 0.01338, None),
    ("propane", "lowP", 450, 1000, 1.2034e-05, 0.038201, None),
    ("propane", "lowP", 550, 1000, 1.4413e-05, 0.054984, None),
    ("propane", "highP", 380, 1e+07, 5.1504e-05, 0.073826, None),
    ("propane", "highP", 580, 1e+07, 1.9274e-05, 0.070186, None),
    # butane: https://webbook.nist.gov/cgi/fluid.cgi?ID=C106978&Action=Data (Type=SatT; IsoBar P=0.001 MPa and 5-10 MPa)
    ("n-butane", "sat", 290.4, 189800, 0.00017163, 0.10789, 0.012807),
    ("n-butane", "sat", 372.9, 1.5184e+06, 7.8662e-05, 0.078301, 0.0040724),
    ("n-butane", "sat", 411.58, 3.0368e+06, 4.6512e-05, 0.067586, 0.00079672),
    ("n-butane", "sat", 422, 3.6062e+06, 3.455e-05, 0.069346, 0.00013505),
    ("n-butane", "lowP", 280, 1000, 6.9653e-06, 0.014842, None),
    ("n-butane", "lowP", 380, 1000, 9.3782e-06, 0.025657, None),
    ("n-butane", "lowP", 480, 1000, 1.1728e-05, 0.039679, None),
    ("n-butane", "highP", 430, 5e+06, 4.1365e-05, 0.06656, None),
    ("n-butane", "highP", 530, 5e+06, 1.5584e-05, 0.054794, None),
    # pentane: https://webbook.nist.gov/cgi/fluid.cgi?ID=C109660&Action=Data (Type=SatT; IsoBar P=0.001 MPa and 5-10 MPa)
    ("n-pentane", "sat", 324.96, 168380, 0.00017434, 0.10238, 0.012538),
    ("n-pentane", "sat", 413.78, 1.347e+06, 8.1999e-05, 0.077489, 0.0037756),
    ("n-pentane", "sat", 455.2, 2.694e+06, 4.8208e-05, 0.071326, 0.0006586),
    ("n-pentane", "sat", 466.37, 3.1991e+06, 3.485e-05, 0.077072, 9.7772e-05),
    ("n-pentane", "lowP", 300, 1000, 6.7905e-06, 0.014601, None),
    ("n-pentane", "lowP", 500, 1000, 1.1278e-05, 0.039861, None),
    ("n-pentane", "lowP", 600, 1000, 1.3319e-05, 0.055407, None),
    ("n-pentane", "highP", 480, 5e+06, 4.2197e-05, 0.071971, None),
    ("n-pentane", "highP", 580, 5e+06, 1.5121e-05, 0.060873, None),
    # hexane: https://webbook.nist.gov/cgi/fluid.cgi?ID=C110543&Action=Data (Type=SatT; IsoBar P=0.001 MPa and 5-10 MPa)
    ("n-hexane", "sat", 355.42, 152210, 0.00017853, 0.10428, 0.012081),
    ("n-hexane", "sat", 449.28, 1.2176e+06, 8.2576e-05, 0.084785, 0.0036689),
    ("n-hexane", "sat", 492.71, 2.4353e+06, 5.0069e-05, 0.077553, 0.0006424),
    ("n-hexane", "sat", 504.36, 2.8919e+06, 3.8027e-05, 0.08024, 8.4547e-05),
    ("n-hexane", "lowP", 350, 1000, 7.376e-06, 0.01849, None),
    ("n-hexane", "lowP", 450, 1000, 9.4371e-06, 0.029383, None),
    ("n-hexane", "lowP", 550, 1000, 1.1442e-05, 0.042074, None),
    ("n-hexane", "highP", 520, 5e+06, 4.8058e-05, 0.0771, None),
    ("n-hexane", "highP", 570, 5e+06, 2.3074e-05, 0.061644, None),
    # heptane: https://webbook.nist.gov/cgi/fluid.cgi?ID=C142825&Action=Data (Type=SatT; IsoBar P=0.001 MPa and 5-10 MPa)
    ("n-heptane", "sat", 382.11, 136790, 0.00018031, 0.098137, 0.011716),
    ("n-heptane", "sat", 479.67, 1.0943e+06, 8.2971e-05, 0.074604, 0.0035452),
    ("n-heptane", "sat", 524.55, 2.1886e+06, 5.1647e-05, 0.065007, 0.000637),
    ("n-heptane", "sat", 536.6, 2.5989e+06, 4.0244e-05, 0.066835, 9.6105e-05),
    ("n-heptane", "lowP", 380, 1000, 7.4467e-06, 0.019348, None),
    ("n-heptane", "lowP", 480, 1000, 9.3914e-06, 0.030354, None),
    ("n-heptane", "lowP", 580, 1000, 1.131e-05, 0.042639, None),
    ("n-heptane", "highP", 550, 5e+06, 5.4596e-05, 0.068315, None),
    ("n-heptane", "highP", 600, 5e+06, 2.9821e-05, 0.063617, None),
    # benzene: https://webbook.nist.gov/cgi/fluid.cgi?ID=C71432&Action=Data (Type=SatT; IsoBar P=0.001 MPa and 5-10 MPa)
    ("benzene", "sat", 385.7, 249910, 0.00023777, 0.11403, 0.017086),
    ("benzene", "sat", 493.41, 1.9658e+06, 9.8801e-05, 0.082599, 0.0052501),
    ("benzene", "sat", 544.24, 3.9268e+06, 5.8701e-05, 0.069383, 0.00093569),
    ("benzene", "sat", 557.93, 4.6622e+06, 4.4097e-05, 0.073765, 0.0001196),
    ("benzene", "lowP", 350, 1000, 8.8658e-06, 0.014888, None),
    ("benzene", "lowP", 550, 1000, 1.379e-05, 0.035484, None),
    ("benzene", "lowP", 650, 1000, 1.625e-05, 0.04647, None),
    ("benzene", "highP", 580, 5e+06, 1.8251e-05, 0.053641, None),
    ("benzene", "highP", 680, 5e+06, 1.8388e-05, 0.055565, None),
    # toluene: https://webbook.nist.gov/cgi/fluid.cgi?ID=C108883&Action=Data (Type=SatT; IsoBar P=0.001 MPa and 5-10 MPa)
    ("toluene", "sat", 410.84, 206320, 0.00020548, 0.10063, 0.014935),
    ("toluene", "sat", 522.28, 1.6505e+06, 9.9161e-05, 0.080292, 0.0043411),
    ("toluene", "sat", 573.76, 3.301e+06, 5.9125e-05, 0.076088, 0.00075855),
    ("toluene", "sat", 587.58, 3.92e+06, 4.3227e-05, 0.081238, 0.00011483),
    ("toluene", "lowP", 400, 1000, 9.2427e-06, 0.019451, None),
    ("toluene", "lowP", 600, 1000, 1.3716e-05, 0.041122, None),
    ("toluene", "lowP", 700, 1000, 1.5891e-05, 0.053011, None),
    ("toluene", "highP", 600, 5e+06, 4.5581e-05, 0.076849, None),
    ("toluene", "highP", 700, 5e+06, 1.8164e-05, 0.058546, None),
    # methanol: https://webbook.nist.gov/cgi/fluid.cgi?ID=C67561&Action=Data (Type=SatT; IsoBar P=0.001 MPa and 5-10 MPa)
    ("methanol", "sat", 377.52, 405180, 0.00021324, 0.18487, 0.015178),
    ("methanol", "sat", 461.89, 3.2414e+06, 9.3612e-05, 0.1694, 0.0058727),
    ("methanol", "sat", 499.63, 6.4828e+06, 5.787e-05, 0.16358, 0.0012885),
    ("methanol", "sat", 509.61, 7.6983e+06, 4.6356e-05, 0.17528, 0.00026032),
    ("methanol", "lowP", 350, 1000, 1.1357e-05, 0.019912, None),
    ("methanol", "lowP", 500, 1000, 1.6423e-05, 0.038048, None),
    ("methanol", "lowP", 600, 1000, 1.981e-05, 0.05257, None),
    ("methanol", "highP", 520, 1e+07, 4.8229e-05, 0.1594, None),
    ("methanol", "highP", 620, 1e+07, 2.2103e-05, 0.077129, None),
    # ammonia: https://webbook.nist.gov/cgi/fluid.cgi?ID=C7664417&Action=Data (Type=SatT; IsoBar P=0.001 MPa and 5-10 MPa)
    ("ammonia", "sat", 281.17, 573920, 0.00015115, 0.50643, 0.024442),
    ("ammonia", "sat", 357.53, 4.549e+06, 7.0898e-05, 0.33253, 0.0077618),
    ("ammonia", "sat", 393.05, 9.0919e+06, 4.3751e-05, 0.24862, 0.0015224),
    ("ammonia", "sat", 402.67, 1.0796e+07, 3.411e-05, 0.24573, 0.00025849),
    ("ammonia", "lowP", 250, 1000, 8.4604e-06, 0.020113, None),
    ("ammonia", "lowP", 500, 1000, 1.7733e-05, 0.051868, None),
    ("ammonia", "lowP", 700, 1000, 2.491e-05, 0.087012, None),
    ("ammonia", "highP", 420, 1e+07, 1.6532e-05, 0.067979, None),
    ("ammonia", "highP", 720, 1e+07, 2.6726e-05, 0.097364, None),
    # r134a: https://webbook.nist.gov/cgi/fluid.cgi?ID=C811972&Action=Data (Type=SatT; IsoBar P=0.001 MPa and 5-10 MPa)
    ("R134a", "sat", 263.49, 203330, 0.00030217, 0.096336, 0.012798),
    ("R134a", "sat", 331.68, 1.6239e+06, 0.00012621, 0.066734, 0.003904),
    ("R134a", "sat", 363.2, 3.2475e+06, 7.2357e-05, 0.052736, 0.00072967),
    ("R134a", "sat", 371.68, 3.8563e+06, 5.3164e-05, 0.053338, 0.00011751),
    ("R134a", "lowP", 260, 1000, 1.0379e-05, 0.010301, None),
    ("R134a", "lowP", 360, 1000, 1.4201e-05, 0.018311, None),
    ("R134a", "lowP", 410, 1000, 1.6071e-05, 0.022316, None),
    ("R134a", "highP", 380, 5e+06, 5.4113e-05, 0.048858, None),
    ("R134a", "highP", 430, 5e+06, 2.0946e-05, 0.028521, None),
    # h2s: https://webbook.nist.gov/cgi/fluid.cgi?ID=C7783064&Action=Data (Type=SatT; IsoBar P=0.001 MPa and 5-10 MPa)
    ("hydrogen sulfide", "sat", 248.1, 457130, 0.00019531, 0.20217, 0.020979),
    ("hydrogen sulfide", "sat", 323.78, 3.599e+06, 9.6773e-05, 0.12632, 0.0068257),
    ("hydrogen sulfide", "sat", 360.15, 7.1897e+06, 6.0846e-05, 0.093936, 0.0013585),
    ("hydrogen sulfide", "sat", 370.05, 8.5362e+06, 4.629e-05, 0.095479, 0.00023729),
    ("hydrogen sulfide", "lowP", 250, 1000, 1.0275e-05, 0.011448, None),
    ("hydrogen sulfide", "lowP", 550, 1000, 2.3314e-05, 0.030208, None),
    ("hydrogen sulfide", "lowP", 750, 1000, 3.1566e-05, 0.045035, None),
    ("hydrogen sulfide", "highP", 380, 1e+07, 2.9603e-05, 0.08383, None),
    ("hydrogen sulfide", "highP", 680, 1e+07, 2.9089e-05, 0.05153, None),
]
KEYS = {"sat": ("mu_liquid_Pa_s", "k_liquid_W_mK", "sigma_N_m"), "lowP": ("mu_gas_Pa_s", "k_gas_W_mK", None),
        "highP": ("mu_gas_at_P_Pa_s", "k_gas_at_P_W_mK", None)}
CLASS = {"nitrogen": "permanent", "oxygen": "permanent", "argon": "permanent", "hydrogen": "permanent",
         "carbon dioxide": "CO2", "water": "water", "methane": "C1-C4", "ethane": "C1-C4", "propane": "C1-C4",
         "n-butane": "C1-C4", "n-pentane": "C5-C7", "n-hexane": "C5-C7", "n-heptane": "C5-C7", "benzene": "aromatic",
         "toluene": "aromatic", "methanol": "alcohol", "ammonia": "NH3", "R134a": "R134a", "hydrogen sulfide": "H2S"}


def _dumps(r: dict) -> None:
    json.dumps(r)


@pytest.fixture(scope="module")
def nist_eval():
    """Tool value against every NIST row: (compound, kind, T, key, deviation, value dict, warnings, near_critical)."""
    out = []
    for name, kind, T, P, *wb in NIST_ROWS:
        r = PU.compound_properties(name, T, P if kind == "highP" else None)
        _dumps(r)
        tc = r["constants"]["Tc_K"]["value"]
        for key, ref in zip(KEYS[kind], wb):
            if key is None or ref is None:
                continue
            p = r["properties_at_T"].get(key) or {}
            fallback = p.get("value") is None and kind == "highP"
            if fallback:
                p = r["properties_at_T"][key.replace("_at_P", "")]      # dilute value: no pressure correction available
            out.append({"name": name, "kind": kind, "T": T, "key": key, "dev": p["value"] / ref - 1.0, "p": p,
                        "warnings": r["warnings"], "fallback": fallback, "near_critical": kind == "sat" and T > 0.95 * tc})
    return out


def test_nist_rows_cover_the_brief():
    names = {r[0] for r in NIST_ROWS}
    assert len(names) == 19 and "water" in names
    for n in names:
        kinds = [r[1] for r in NIST_ROWS if r[0] == n]
        assert kinds.count("sat") == 4 and kinds.count("lowP") >= 2 and kinds.count("highP") >= 1, n


def test_tool_against_nist_by_class(nist_eval):
    """Measured bands of the tool's returned value against NIST WebBook (2026-11-15), per class of value."""
    def worst(sel):
        v = [abs(e["dev"]) for e in nist_eval if sel(e)]
        return max(v), len(v)

    ref = lambda e: e["p"]["method"] == "COOLPROP_REFERENCE" and not e["p"].get("flag") and not e["near_critical"]
    m, n = worst(lambda e: ref(e) and e["kind"] == "sat")
    assert n >= 80 and m < 0.006                       # measured 0.57 % (n-heptane k_liquid 480 K)
    m, n = worst(lambda e: ref(e) and e["kind"] == "lowP")
    assert n >= 70 and m < 0.008                       # measured 0.72 % (methane k_gas 150 K)
    m, n = worst(lambda e: ref(e) and e["kind"] == "highP")
    assert n >= 45 and m < 0.003                       # measured 0.21 % (methane k at 10 MPa, 600 K)
    thermo = lambda e: e["p"]["method"] == "REFPROP_FIT" and not e["near_critical"] and not e["fallback"]
    m, n = worst(lambda e: thermo(e) and e["kind"] == "sat")
    assert n >= 20 and m < 0.002                       # measured 0.13 % (n-pentane mu_liquid 414 K)
    m, n = worst(lambda e: thermo(e) and e["kind"] == "lowP")
    assert n >= 35 and m < 0.025                       # measured 2.26 % (ethane k_gas 200 K), viscosity ≤ 0.79 %
    m, n = worst(lambda e: e["near_critical"] and e["key"] == "mu_liquid_Pa_s")
    assert m < 0.008                                   # measured 0.70 %
    m, n = worst(lambda e: e["near_critical"] and e["key"] == "k_liquid_W_mK")
    assert 0.4 < m < 0.55                              # measured 50.6 % (H2S, VDI_PPDS at T/Tc = 0.99): flagged
    m, n = worst(lambda e: e["near_critical"] and e["key"] == "sigma_N_m")
    assert m < 0.07                                    # measured 6.8 % (ammonia at T/Tc = 0.993)


def test_every_large_deviation_is_flagged_or_warned(nist_eval):
    """Honesty invariant: a value more than 3 % from NIST always carries a flag, or (dilute fallback) a warning."""
    big = [e for e in nist_eval if abs(e["dev"]) > 0.03]
    assert len(big) >= 15
    for e in big:
        if e["fallback"]:
            assert any(e["key"] in w and "diluido" in w for w in e["warnings"]), e
        else:
            assert e["p"].get("flag"), (e["name"], e["T"], e["key"], e["dev"])
    for e in nist_eval:
        if e["near_critical"]:
            assert "punto crítico" in e["p"]["flag"]


def test_class_table_for_the_report(nist_eval):
    """Per compound class and property, max |tool / NIST - 1| outside the near-critical zone (printed with -s)."""
    table: dict = {}
    for e in nist_eval:
        if not e["near_critical"]:
            k = (CLASS[e["name"]], e["key"])
            table[k] = max(table.get(k, 0.0), abs(e["dev"]))
    for k in sorted(table):
        print(f"{k[0]:10s} {k[1]:18s} {100 * table[k]:6.2f} %")
    assert table[("permanent", "mu_liquid_Pa_s")] < 0.001 and table[("aromatic", "k_gas_at_P_W_mK")] < 0.002
    assert table[("C5-C7", "mu_gas_at_P_Pa_s")] > 0.4          # n-pentane: CoolProp's friction theory, flagged
    assert table[("NH3", "k_gas_at_P_W_mK")] > 0.1             # ammonia: Tufeu 1984 in CoolProp, flagged


def test_coolprop_correlations_against_nist_table_is_what_was_measured():
    """The constants _CP_VS_NIST / _CP_SIGMA_VS_NIST used by the tool are reproduced from the NIST rows."""
    worst: dict = {}
    for name, kind, T, P, mu, k, s in NIST_ROWS:
        meta, _ = PU.identify(name)
        fl = PU._coolprop_fluid(meta.CASs)
        for what, ref in (("mu", mu), ("k", k), ("sigma", s)):
            if ref is None or fl["kinds"][what][0] != "reference":
                continue
            v = PU._coolprop_value(fl, what, "liquid" if kind == "sat" else "gas", T, P)
            if v is None:
                continue
            slot = (fl["cas"], what, what == "sigma" and T > PU.NEAR_CRITICAL_TR * fl["Tc"])
            worst[slot] = max(worst.get(slot, 0.0), abs(v / ref - 1.0))
    for (cas, what, near), dev in worst.items():
        name = PU._coolprop_fluid(cas)["name"]
        if what == "sigma":
            assert dev <= PU._CP_SIGMA_VS_NIST[cas][1 if near else 0], (name, near, dev)
        else:
            assert dev <= PU._CP_VS_NIST[cas][0 if what == "mu" else 1], (name, what, dev)
    assert worst[("109-66-0", "mu", False)] > 0.4 and worst[("7664-41-7", "k", False)] > 0.1      # n-pentane, ammonia
    assert worst[("7727-37-9", "mu", False)] < 1e-3 and worst[("108-88-3", "k", False)] < 3e-3     # nitrogen, toluene


# ---- against CoolProp's reference correlations: 20 compounds, 3 liquid and 3 gas temperatures each -----------------------
CP_FLUIDS = [("nitrogen", "Nitrogen"), ("oxygen", "Oxygen"), ("argon", "Argon"), ("carbon dioxide", "CarbonDioxide"),
             ("methane", "Methane"), ("ethane", "Ethane"), ("propane", "Propane"), ("n-butane", "n-Butane"),
             ("n-pentane", "n-Pentane"), ("n-hexane", "n-Hexane"), ("n-heptane", "n-Heptane"), ("toluene", "Toluene"),
             ("benzene", "Benzene"), ("methanol", "Methanol"), ("ethanol", "Ethanol"), ("ammonia", "Ammonia"),
             ("R134a", "R134a"), ("hydrogen sulfide", "HydrogenSulfide"), ("isobutane", "IsoButane"),
             ("o-xylene", "o-Xylene")]


def test_tool_against_coolprop_reference_correlations():
    """Consistency: when the tool says COOLPROP_REFERENCE its value IS CoolProp's correlation (independent PropsSI
    call); otherwise reference_check carries that correlation and any disagreement beyond 2 % is flagged."""
    n_used, n_checked, dev_thermo = 0, 0, {}
    for name, cpn in CP_FLUIDS:
        tt, tc, tmax = CP.PropsSI("T_triple", cpn), CP.PropsSI("Tcrit", cpn), CP.PropsSI("Tmax", cpn)
        tl = [tt + f * (tc - tt) for f in (0.1, 0.45, 0.8)]
        tg = [t for t in (0.7 * tc, 1.2 * tc, 1.8 * tc) if t <= tmax]
        for kind, temps in (("liquid", tl), ("gas", tg)):
            for T in temps:
                r = PU.compound_properties(name, T)
                _dumps(r)
                keys = (("mu_liquid_Pa_s", "V"), ("k_liquid_W_mK", "L"), ("sigma_N_m", "I")) if kind == "liquid" \
                    else (("mu_gas_Pa_s", "V"), ("k_gas_W_mK", "L"))
                for key, code in keys:
                    try:
                        ref = (CP.PropsSI(code, "T", T, "Q", 0, cpn) if kind == "liquid"
                               else CP.PropsSI(code, "T", T, "Dmolar", 1e-6, cpn))
                    except ValueError:
                        continue                                    # CoolProp has no correlation (e.g. H2S k)
                    p = r["properties_at_T"][key]
                    if p["method"] == "COOLPROP_REFERENCE":
                        assert abs(p["value"] / ref - 1) < 1e-9, (name, T, key)
                        n_used += 1
                    elif "reference_check" not in p:
                        # no CoolProp check: no reference correlation, or T below CoolProp's EOS range (dilute gas)
                        assert p.get("reference_correlation", "x") is None or T < CP.PropsSI("Tmin", cpn), (name, T, key)
                    else:
                        chk = p["reference_check"]
                        assert abs(chk["value"] / ref - 1) < 1e-9
                        assert abs(chk["deviation_rel"] - (p["value"] / ref - 1)) < 1e-9
                        if abs(chk["deviation_rel"]) > PU.TRANSPORT_CHECK_TOL:
                            assert "correlación de referencia" in p["flag"], (name, T, key)
                        dev_thermo[(name, key)] = max(dev_thermo.get((name, key), 0.0), abs(chk["deviation_rel"]))
                        n_checked += 1
    assert n_used >= 150 and n_checked >= 50
    # measured |thermo / CoolProp - 1| where thermo's value is kept (CoolProp off NIST or not compared with NIST):
    assert dev_thermo[("n-pentane", "mu_liquid_Pa_s")] > 0.2          # CoolProp is the one off (NIST rows above)
    assert dev_thermo[("ethanol", "mu_liquid_Pa_s")] < 0.003          # Kiselev 2005 vs thermo REFPROP_FIT: measured 0.07 %
    assert dev_thermo[("ethanol", "k_gas_W_mK")] < 0.012              # measured 1.08 % (at 1.8·Tc)
    for k in sorted(dev_thermo):
        print(f"thermo vs CoolProp {k[0]:16s} {k[1]:15s} {100 * dev_thermo[k]:7.2f} %")
    print("values equal to CoolProp:", n_used, " cross-checked:", n_checked)


def test_classification_of_coolprop_transport_models():
    kinds = {n: PU._coolprop_fluid(CP.get_fluid_param_string(n, "CAS"))["kinds"]
             for n in ("Toluene", "Water", "Methanol", "Propylene", "Cyclopentane", "R32", "Acetone", "HydrogenSulfide")}
    assert kinds["Toluene"]["mu"] == ("reference", "Avgeri-JPCRD-2015-Toluene")
    assert kinds["Water"]["k"][0] == kinds["Methanol"]["mu"][0] == "reference"        # hard-coded correlations
    assert kinds["Propylene"]["mu"][0] == "ECS" and kinds["Cyclopentane"]["mu"][0] == "Chung"
    assert kinds["R32"]["mu"][0] == "rhosr-CS" and kinds["Acetone"]["mu"][0] == "none"
    assert kinds["HydrogenSulfide"]["mu"][0] == "reference" and kinds["HydrogenSulfide"]["k"][0] == "none"
    p = PU.compound_properties("propylene", 250.0)["properties_at_T"]["mu_liquid_Pa_s"]
    assert p["method"] != "COOLPROP_REFERENCE" and "estados correspondientes (ECS)" in p["note"]
    a = PU.compound_properties("acetone", 300.0)
    _dumps(a)
    pa = a["properties_at_T"]
    assert pa["mu_liquid_Pa_s"]["reference_correlation"] is None and pa["sigma_N_m"]["method"] == "COOLPROP_REFERENCE"


def test_dense_gas_limits_and_monotonicity():
    # zero-pressure limit: the at-P value tends to the dilute one
    r = PU.compound_properties("nitrogen", 300.0, 10.0)
    _dumps(r)
    for key in ("mu_gas_at_P_Pa_s", "k_gas_at_P_W_mK"):
        assert abs(r["properties_at_T"][key]["pressure_effect_rel"]) < 1e-5
    # supercritical N2 at 300 K: viscosity and conductivity rise with density
    mus = [PU.compound_properties("nitrogen", 300.0, P)["properties_at_T"]["mu_gas_at_P_Pa_s"]["value"]
           for P in (1e5, 1e6, 5e6, 1e7, 2e7)]
    assert all(b > a for a, b in zip(mus, mus[1:]))
    w = PU.compound_properties("methane", 300.0, 1e7)
    assert any("mu_gas_at_P_Pa_s" in x and "usa" in x for x in w["warnings"])
    # no reference correlation: dilute only, and the tool says so when the gas is dense
    g = PU.compound_properties("acetone", 450.0, 1.0e6)
    _dumps(g)
    assert g["state"]["phase"] == "gas" and g["properties_at_T"]["mu_gas_at_P_Pa_s"]["value"] is None
    assert any("mu_gas_at_P_Pa_s: no disponible" in x for x in g["warnings"])
    h = PU.compound_properties("hydrogen sulfide", 400.0, 1e7)          # mu has a correlation, k does not
    assert h["properties_at_T"]["mu_gas_at_P_Pa_s"]["value"] and h["properties_at_T"]["k_gas_at_P_W_mK"]["value"] is None
    assert "contra NIST WebBook" in h["properties_at_T"]["mu_gas_at_P_Pa_s"]["flag"]


def test_method_bands_and_flags():
    # thermo method with a wide measured band is flagged with it (H2S liquid conductivity near Tc falls to VDI_PPDS)
    p = PU.compound_properties("hydrogen sulfide", 360.0)["properties_at_T"]["k_liquid_W_mK"]
    assert p["method"] == "VDI_PPDS" and p["measured_band_rel"]["p90"] > PU.METHOD_BAND_FLAG and "10 % peor" in p["flag"]
    # the band table reproduces: REFPROP_FIT liquid viscosity against the NIST-matching CoolProp correlations
    devs = []
    for name, (bm, _) in PU._CP_VS_NIST.items():
        if bm > PU.NIST_AGREE_TOL:
            continue
        fl = PU._coolprop_fluid(CP.get_fluid_param_string(name, "CAS"))
        meta, _ = PU.identify(CP.get_fluid_param_string(name, "CAS"))
        obj = PU._package(meta.CASs)[1].ViscosityLiquids[0]
        for f in (0.05, 0.3, 0.55, 0.8, 0.92):
            T = fl["Tmin"] + f * (fl["Tc"] - fl["Tmin"])
            v = PU._calc(obj, T, "REFPROP_FIT") if "REFPROP_FIT" in obj.T_limits and \
                obj.T_limits["REFPROP_FIT"][0] <= T <= obj.T_limits["REFPROP_FIT"][1] else None
            if v:
                devs.append(abs(v / PU._coolprop_value(fl, "mu", "liquid", T) - 1))
    devs.sort()
    p90, mx, n = PU._METHOD_BAND["mu_liquid_Pa_s"]["REFPROP_FIT"]
    assert len(devs) == n and abs(devs[int(0.9 * (n - 1))] - p90) < 1e-5 and abs(devs[-1] - mx) < 1e-5


def test_water_keeps_iapws_and_old_limit_text_is_gone():
    w = PU.compound_properties("water", 350.0)
    assert w["properties_at_T"]["sigma_N_m"]["method"] == "IAPWS_SIGMA"
    assert w["properties_at_T"]["mu_liquid_Pa_s"]["reference_correlation"] == "Huber-JPCRD-2009"
    old = "Viscosidad y conductividad del gas a alta presión: se dan las del gas diluido."
    assert old not in w["not_included"] and any("mu_gas_at_P_Pa_s" in t for t in w["not_included"])
    assert "test_transport_properties" in TPR.compound_properties.__doc__


def test_refusals_and_wrapper():
    with pytest.raises(ValueError, match="positiva"):
        PU.compound_properties("toluene", 0.0)
    with pytest.raises(ValueError, match="sin T_K"):
        PU.compound_properties("toluene", None, 1e6)
    with pytest.raises(ValueError, match="positiva"):
        PU.compound_properties("toluene", 400.0, -1.0)
    out = TPR.compound_properties("toluene", 600.0, 5e6)
    _dumps(out)
    assert out["properties_at_T"]["mu_gas_at_P_Pa_s"]["method"] == "COOLPROP_REFERENCE"
    assert [getattr(w, "name", None) or w.__name__ for w in WP.AGENT_WRAPPERS][0] == "compound_properties"


def test_independent_of_thermosteam_global_flag():
    """BioSTEAM's thermosteam sets thermo's RAISE_PROPERTY_CALCULATION_ERROR = True globally (found in the full suite,
    2026-11-15): the tool must still answer for R134a (no sublimation-enthalpy method) and restore the flag."""
    from thermo.utils.t_dependent_property import TDependentProperty
    saved = TDependentProperty.RAISE_PROPERTY_CALCULATION_ERROR
    TDependentProperty.RAISE_PROPERTY_CALCULATION_ERROR = True
    try:
        PU._package.cache_clear()
        r = PU.compound_properties("R134a", 250.0)
        assert r["properties_at_T"]["mu_liquid_Pa_s"]["value"] > 0
        assert TDependentProperty.RAISE_PROPERTY_CALCULATION_ERROR is True
    finally:
        TDependentProperty.RAISE_PROPERTY_CALCULATION_ERROR = saved
