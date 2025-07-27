#--------------------------------------------------------
# Functions that take in the photochemical model results
#--------------------------------------------------------
import numpy as np
import pandas as pd
from math import floor
EARTH_RAD = 6.37E6 #radius of Earth [m]
def atmospheric_density_and_oxygen(altitude, co2_surface_vol, co2_lower, co2_upper):
    """
    This function will return the atmospheric density and the total oxygen
    density (including atomic O and O2) for the given altitude. This is done
    using data from the MSISE-90 model. The data is in 1km steps, which this
    function will linearly interpolate between. The data arrays come from the
    supplental data file atmosphere_data.txt. The arrays were hardcoded below
    to avoid reading them in during each model run.

    Inputs:
        altitude - the altitude above the Earth's center [m]
        co2_surface_vol - wt percent of the surface CO2 concentration
        co2_lower - co2_lower file 
        co2_upper - co2_upper file
    Returns:
        rho_a - atmospheric density [kg m-3]
        rho_o - total oxygen(O and O2) density [kg m-3]
        co2_wt - total co2 density
    """
    
    alt = (altitude - EARTH_RAD)/1000 #convert to traditional altitude [km]
    rho_a = 0
    #data from atmosphere_data.txt
    alt_data = [0.00e+00, 1.00e+00, 2.00e+00, 3.00e+00, 4.00e+00, 5.00e+00, 
                6.00e+00, 7.00e+00, 8.00e+00, 9.00e+00, 1.00e+01, 1.10e+01, 
                1.20e+01, 1.30e+01, 1.40e+01, 1.50e+01, 1.60e+01, 1.70e+01, 
                1.80e+01, 1.90e+01, 2.00e+01, 2.10e+01, 2.20e+01, 2.30e+01, 
                2.40e+01, 2.50e+01, 2.60e+01, 2.70e+01, 2.80e+01, 2.90e+01, 
                3.00e+01, 3.10e+01, 3.20e+01, 3.30e+01, 3.40e+01, 3.50e+01, 
                3.60e+01, 3.70e+01, 3.80e+01, 3.90e+01, 4.00e+01, 4.10e+01, 
                4.20e+01, 4.30e+01, 4.40e+01, 4.50e+01, 4.60e+01, 4.70e+01, 
                4.80e+01, 4.90e+01, 5.00e+01, 5.10e+01, 5.20e+01, 5.30e+01, 
                5.40e+01, 5.50e+01, 5.60e+01, 5.70e+01, 5.80e+01, 5.90e+01, 
                6.00e+01, 6.10e+01, 6.20e+01, 6.30e+01, 6.40e+01, 6.50e+01,
                6.60e+01, 6.70e+01, 6.80e+01, 6.90e+01, 7.00e+01, 7.10e+01, 
                7.20e+01, 7.30e+01, 7.40e+01, 7.50e+01, 7.60e+01, 7.70e+01, 
                7.80e+01, 7.90e+01, 8.00e+01, 8.10e+01, 8.20e+01, 8.30e+01, 
                8.40e+01, 8.50e+01, 8.60e+01, 8.70e+01, 8.80e+01, 8.90e+01, 
                9.00e+01, 9.10e+01, 9.20e+01, 9.30e+01, 9.40e+01, 9.50e+01, 
                9.60e+01, 9.70e+01, 9.80e+01, 9.90e+01, 1.00e+02, 1.01e+02, 
                1.02e+02, 1.03e+02, 1.04e+02, 1.05e+02, 1.06e+02, 1.07e+02, 
                1.08e+02, 1.09e+02, 1.10e+02, 1.11e+02, 1.12e+02, 1.13e+02, 
                1.14e+02, 1.15e+02, 1.16e+02, 1.17e+02, 1.18e+02, 1.19e+02, 
                1.20e+02, 1.21e+02, 1.22e+02, 1.23e+02, 1.24e+02, 1.25e+02,
                1.26e+02, 1.27e+02, 1.28e+02, 1.29e+02, 1.30e+02, 1.31e+02, 
                1.32e+02, 1.33e+02, 1.34e+02, 1.35e+02, 1.36e+02, 1.37e+02,
                1.38e+02, 1.39e+02, 1.40e+02, 1.41e+02, 1.42e+02, 1.43e+02,
                1.44e+02, 1.45e+02, 1.46e+02, 1.47e+02, 1.48e+02, 1.49e+02,
                1.50e+02, 1.51e+02, 1.52e+02, 1.53e+02, 1.54e+02, 1.55e+02,
                1.56e+02, 1.57e+02, 1.58e+02, 1.59e+02, 1.60e+02, 1.61e+02,
                1.62e+02, 1.63e+02, 1.64e+02, 1.65e+02, 1.66e+02, 1.67e+02, 
                1.68e+02, 1.69e+02, 1.70e+02, 1.71e+02, 1.72e+02, 1.73e+02,
                1.74e+02, 1.75e+02, 1.76e+02, 1.77e+02, 1.78e+02, 1.79e+02,
                1.80e+02, 1.81e+02, 1.82e+02, 1.83e+02, 1.84e+02, 1.85e+02,
                1.86e+02, 1.87e+02, 1.88e+02, 1.89e+02, 1.90e+02]
    
    rho_a_data = [1.21e+00, 1.09e+00, 9.84e-01, 8.90e-01, 8.05e-01, 7.28e-01, 
                  6.57e-01, 5.90e-01, 5.28e-01, 4.70e-01, 4.15e-01, 3.65e-01, 
                  3.19e-01, 2.77e-01, 2.39e-01, 2.06e-01, 1.76e-01, 1.50e-01, 
                  1.28e-01, 1.09e-01, 9.23e-02, 7.86e-02, 6.69e-02, 5.70e-02, 
                  4.86e-02, 4.15e-02, 3.55e-02, 3.03e-02, 2.59e-02, 2.22e-02, 
                  1.90e-02, 1.63e-02, 1.39e-02, 1.20e-02, 1.03e-02, 8.84e-03, 
                  7.63e-03, 6.60e-03, 5.72e-03, 4.97e-03, 4.32e-03, 3.77e-03, 
                  3.30e-03, 2.89e-03, 2.54e-03, 2.24e-03, 1.98e-03, 1.75e-03, 
                  1.55e-03, 1.37e-03, 1.22e-03, 1.08e-03, 9.60e-04, 8.53e-04, 
                  7.58e-04, 6.74e-04, 5.98e-04, 5.31e-04, 4.71e-04, 4.17e-04,
                  3.69e-04, 3.26e-04, 2.88e-04, 2.53e-04, 2.23e-04, 1.96e-04, 
                  1.71e-04, 1.50e-04, 1.31e-04, 1.14e-04, 9.88e-05, 8.57e-05,
                  7.41e-05, 6.37e-05, 5.50e-05, 4.74e-05, 4.08e-05, 3.51e-05, 
                  3.00e-05, 2.55e-05, 2.16e-05, 1.82e-05, 1.53e-05, 1.27e-05,
                  1.05e-05, 8.67e-06, 7.10e-06, 5.79e-06, 4.69e-06, 3.79e-06, 
                  3.05e-06, 2.45e-06, 1.96e-06, 1.57e-06, 1.26e-06, 1.01e-06,
                  8.20e-07, 6.65e-07, 5.43e-07, 4.47e-07, 3.70e-07, 3.09e-07, 
                  2.60e-07, 2.21e-07, 1.88e-07, 1.61e-07, 1.38e-07, 1.19e-07,
                  1.03e-07, 8.90e-08, 7.70e-08, 6.66e-08, 5.77e-08, 4.99e-08, 
                  4.33e-08, 3.76e-08, 3.27e-08, 2.86e-08, 2.50e-08, 2.19e-08,
                  1.93e-08, 1.70e-08, 1.51e-08, 1.34e-08, 1.20e-08, 1.08e-08, 
                  9.76e-09, 8.87e-09, 8.09e-09, 7.41e-09, 6.81e-09, 6.28e-09,
                  5.81e-09, 5.38e-09, 5.00e-09, 4.66e-09, 4.36e-09, 4.08e-09, 
                  3.82e-09, 3.59e-09, 3.38e-09, 3.18e-09, 3.00e-09, 2.84e-09,
                  2.69e-09, 2.55e-09, 2.42e-09, 2.30e-09, 2.18e-09, 2.08e-09, 
                  1.98e-09, 1.89e-09, 1.81e-09, 1.72e-09, 1.65e-09, 1.58e-09,
                  1.51e-09, 1.45e-09, 1.39e-09, 1.33e-09, 1.28e-09, 1.23e-09, 
                  1.18e-09, 1.14e-09, 1.10e-09, 1.06e-09, 1.02e-09, 9.82e-10,
                  9.47e-10, 9.14e-10, 8.83e-10, 8.53e-10, 8.24e-10, 7.96e-10, 
                  7.70e-10, 7.45e-10, 7.21e-10, 6.98e-10, 6.76e-10, 6.55e-10,
                  6.35e-10, 6.16e-10, 5.97e-10, 5.79e-10, 5.62e-10, 5.46e-10, 
                  5.30e-10, 5.14e-10, 5.00e-10, 4.86e-10, 4.72e-10]

    ratio = (co2_upper-co2_surface_vol)/(co2_upper-co2_lower) #get the ratio that for lower ones directly

    # read the file
    read_co2 = './photochem/photochem_180K_1e5_CH4low/' #select inputting photochemical folder/files
    co2_lower_file = pd.read_csv(f'{read_co2}photochem_{co2_lower}.csv')
    co2_upper_file = pd.read_csv(f'{read_co2}photochem_{co2_upper}.csv')
    
    # mixing ratio of CO2 and O
    co2_value = co2_lower_file['CO2'] * ratio + co2_upper_file['CO2'] * (1 - ratio)
    o_value   = co2_lower_file['O']   * ratio + co2_upper_file['O'] * (1 - ratio) + 2*(co2_lower_file['O2'] * (ratio) + co2_upper_file['O2'] * (1-ratio))
    
    # make calculate the mass density
    co2_value_wt = co2_value*44/(co2_value*44+o_value*16+(1-co2_value-o_value)*28)
    o_value_wt = o_value*16/(co2_value*44+o_value*16+(1-co2_value-o_value)*28)
    
    if alt < alt_data[0] or alt > alt_data[-1]:
        raise ValueError(f"altitude {alt} km out of bounds ({alt_data[0]} - {alt_data[-1]} km)")

    o_wt = 0
    co2_wt = 0
    
    if 100 <= alt < 190:
        idx = int(floor(alt))
        frac_low = 1 - (alt-alt_data[idx])/(alt_data[idx+1] - alt_data[idx])
        rho_a = rho_a_data[idx]*frac_low + rho_a_data[idx+1]*(1-frac_low)
    elif 0 < alt <= 100:
        idx = int(floor(alt))
        frac_low = 1 - (alt-alt_data[idx])/(alt_data[idx+1] - alt_data[idx])
        rho_a = rho_a_data[idx]*frac_low + rho_a_data[idx+1]*(1-frac_low)
        o_wt = o_value_wt[idx]*frac_low+ o_value_wt[idx+1]*(1-frac_low)
        co2_wt = co2_value_wt[idx]*frac_low+ co2_value_wt[idx+1]*(1-frac_low)
    return rho_a, o_wt, co2_wt