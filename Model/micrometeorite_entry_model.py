#-------------------------------------------------------------------------------------------
# This model was originally developed by Dr Owen Lehmer and first published in:
# Lehmer, O.R., Catling, D.C., Buick, R., Brownlee, D.E., & Newport, S. (2020). 
# Atmospheric CO₂ levels from 2.7 billion years ago inferred from micrometeorite oxidation. 
# *Science Advances*, 6(4), eaay4644. https://doi.org/10.1126/sciadv.aay4644
#
# For a detailed description of the original model, please refer to the publication above.
# This version has been modified by Danqiu Chen. For inquiries, please contact: chendq@uw.edu
# -------------------------------------------------------------------------------------------
from multiprocessing import Pool, cpu_count
from math import sin, cos, asin, pi, floor, ceil, sqrt, exp
from scipy import stats
from tqdm import tqdm
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from atmospheric_cal import atmospheric_density_and_oxygen
from contextlib import redirect_stdout

#################constants defined here###############
EARTH_RAD = 6.37E6 #radius of Earth [m]
EARTH_MASS = 5.97E24 #Earth mass [kg]
GRAV_CONST = 6.67E-11 #gravitational constant [N m2 kg-2]
KB = 1.381E-23 #Boltzmann constant [J K-1]
PROTON_MASS = 1.67E-27 #mass of a proton [kg]
SIGMA = 5.67E-8 #Stefan-Boltzmann constant [W m-2 K-4]
GAS_CONST = 8.314 #ideal gas constant [J mol-1 K-1]
M_FE = 0.0558 #molecular mass of Fe [kg mol-1]
M_O = 0.016 #molecular mass of O [kg mol-1]
M_FEO = M_FE + M_O #molecular mass of FeO [kg mol-1]
M_CO2 = 0.044 #molecular mass of CO2 [kg mol-1]
M_N2 = 0.028 #molecular mass of N2 [kg mol-1]
DELTA_H_OX_CO2 = -465000 #heat of oxidation for CO2 + Fe -> CO +FeO [J kg-1]
DELTA_H_OX_O2 = 3716000 #heat of oxidation for O+Fe->FeO [J kg-1] 
########TEMP THRESHOLD #######
temp_thre = 1650 #reaction temperature threshold [K]

#the latent heat is the same for Fe and FeO in this model
L_V = 6.0E6 #latent heat of vaporization for FeO/Fe [J kg-1]
#the specific heat is roughly the same for FeO and Fe (and has very little
#impact on the calculations within a factor of a few)
C_SP = 400 #specific heat of FeO from Stolen et al. (2015) [J K-1 kg-1]
FE_REACTION_TEMP = 1000 #temperature at which Fe reacts with CO2
FE_MELTING_TEMP = 1809 #temperature at which Fe melts [K]
FEO_MELTING_TEMP = 1720 #melting temp of FeO [K]

#densities of the two liquids considered in this model
RHO_FE = 7000 #liquid Fe density [kg m-3]
RHO_FEO = 4400 #liquid FeO density [kg m-3]

GAMMA = 1.0 #scaling parameter in eq. 10 of Lehmer+2020

ADD_OX_EST = False #set to true to add 1% O2 to the atmosphere
######################end constants###############################

class impactAngleDistribution(stats.rv_continuous):
    """
    The probability distribution of impact angles. This class will generate
    random entry angles from the distribution defined in Love and Brownlee 
    (1991).
    """

    def __init__(self):
        """
        Set the lower limit to 0 degrees and the upper limit to 90 degrees
        """
        super().__init__(a=0, b=pi/2)

    def _pdf(self, x):
        prob = 0
        if 0 < x < pi/2:
            #between 0 and 90 degrees
            prob = sin(2*x)
        return prob

    def sample(self, size=1, random_state=None):
        return self.rvs(size=size, random_state=random_state)

class initialVelocityDistribution(stats.rv_continuous):
    """
    The probability distribution of initial velocity. This class will generate
    random initial velocities from the distribution defined in Love and Brownlee 
    (1991).

    Note: the values generated from sample are in [km s-1]

    Second note: the velocity is capped at 20 km/s. This is done because faster
    entries require smaller time steps, which makes the model unreasonably
    slow. This is likely acceptable as velocities under 20 km/s account for 
    92.2 percent of the incident velocities, so the final result should be 
    representative of the distribution.
    """

    def __init__(self):
        """
        Set the lower limit to 11.2 [km s-1] and the upper to 20 [km s-1]
        """
        super().__init__(a=11.2, b=20.0) #upper limit set to 20 km s-1


    def _pdf(self, x):
        prob = 0
        if x > 11.2:
            prob = 1.791E5*x**-5.394
        return prob

    def sample(self, size=1, random_state=None):
        return self.rvs(size=size, random_state=random_state)


class initialMassDistribution(stats.rv_continuous):
    """
    The probability distribution of initial mass. This class will generate
    random initial masses from the distribution defined in Love and Brownlee 
    (1991). The masses can easily be converted to initial radii.

    Note: the values generated from the PDF are in [g]
    """

    def __init__(self):
        """
        Set the lower limit to 2.346E-10 [g] (2 [micron] Fe radius) and the upper 
        2.35e-4 [g] (250 [micron] Fe radius).
        """
        super().__init__(a=2.346E-10, b=2.35e-4)


    def _pdf(self, x):
        prob = 0
        if 2.346E-10  < x < 2.35e-4: 
            prob = ((2.2E3*x**0.306+15)**-4.38 + 1.3E-9*(x + 10**11*x**2 + \
                    10**27*x**4)**-0.36)/4.59811E-13
            prob = float(prob)
        return prob

    def sample(self, size=1, random_state=None):
        return self.rvs(size=size, random_state=random_state)




def find_nearest_bounds(value, lst):
    lst_sorted = sorted(lst)
    lower = max([x for x in lst_sorted if x <= value], default=None)
    upper = min([x for x in lst_sorted if x >= value], default=None)
    return lower, upper



def get_radius_and_density(m_fe, m_feo, not_array=True):
    """
    Calculate the radius and bulk density of the micrometeorite using the mass
    fraction of Fe to FeO.

    Inputs:
        M_Fe  - mass of Fe in the micrometeorite [kg]
        M_FeO - mass of FeO in the micrometeorite [kg]

    Returns:
        new_rad - updated micrometeorite radius [m]
        new_rho - updated micrometeorite density [kg m-3]
    """

    if not_array and m_fe + m_feo <= 0:
        return 0, 0
    volume = m_fe/RHO_FE + m_feo/RHO_FEO
    new_rad = (3*volume/(4*pi))**(1/3)

    new_rho = (m_fe + m_feo)/volume

    return new_rad, new_rho


def fe_co2_rate_constant(temp):
    """
    Calculate the rate constant for the reaction Fe + CO2 -> FeO + CO from
    Smirnov (2008). The reaction is assumed to be first order.

    Inputs:
        temp - the temperature of the micrometeorite [K]

    Returns:
        k_fe_co2 - the reaction rate constant [m3 mol-1 s-1]
    """

    #k_fe_co2 = 2.9E8*exp(-15155/temp)
    k_fe_co2 = 268*np.exp(-224115.4/(8.314*temp)) #[mol m-2 s-1 Pa-1]
    return k_fe_co2

def fe_co2_rate_pressure(temp, ram_pressure):
    '''
    input:
        temp: temperature in K
        ram_pressure: in Pa
    output:
        k_fe_co2: in mol/[Pa s m2]
        A: pre_exponential factor, unit same as k_fe_co2
    '''
    if ram_pressure <0.001:
        return 0,0
    
    A = 6635*(ram_pressure/100)**(-1.22)
    Ea =  209166 #(-8.42*np.log(ram_pressure/100) + 244.9)*1e3
    k_fe_co2 = A * np.exp(-Ea / (8.314 * temp))
    return k_fe_co2, A

def dynamic_ode_solver(func, start_time, max_time, initial_guess, 
        param_to_monitor, max_param_delta,
        base_time_step, min_step_time, end_condition):
    """
    Solves a system of ordinary differential equations with dynamic time steps.
    The passed in function (func) should take two arguments, a time step value,
    and the current parameter values. It will have the form:
        func(time_step, ys)
    The func provided should return the derivatives of each value considered.
    This routine will start at time start_time, and run until max_time is 
    reached, or the end condition is met. The end condition is set by the
    end_condition paramter, which should be a function like func that takes
    both the current time (NOT TIME STEP) and the system values like:
        end_condition(current_time, ys)
    and should return not 0 if the integration should terminate, or 0 if it 
    should stop. 
    This solver will attempt each time step and if the % change in the monitored 
    parameter is greater than the provided max_param_delta (as a fraction), then
    the time step will be reduced by half (repeatedly if necessary) until the
    minimum step time is reached (min_step_time). The step time will slowly
    relax back to the base time step (base_time_step) specified by the input. 
    The parameter to monitor should be an integer in the ys array. For example,
    if the initial guess has values:
        initial_guess = [position, velocity]
    and you want to make sure the position never changes by more than 1% you'd
    set param_to_monitor=0 and max_param_delta=0.01. 

    Inputs:
        func             - function that takes time step and values, returns 
                           derivatives
        start_time       - simulation start time [s]
        max_time         - maximum time to run simulation before returning an 
                           error [s]
        initial_guess    - array with initial parameter values
        param_to_monitor - index of which parameter to track
        max_param_delta  - fractional difference to tolerate in param_to_monitor. 
                           A value of 0.01 means changes must be less than 1%. 
                           A value of 0.2 would mean values must be less than 20%.
        base_time_step   - the default step size to use in integration [s]
        min_step_time    - the smallest time step to allow. If max_param_delta 
                           is exceeded at the smallest allowed time step an 
                           error status will be returned.
        end_condition    - function that takes current time and values, returns 
                           not 0 if the integration should end, 0 otherwise. The
                           value of end condition will be passed out of this 
                           function.

    Returns:
        times  - the array of time values [s] calculated 
        ys     - the array of parameter values at each time step
        status - the status of the solver. Values are:
                    -1 : failure because max_param_delta was exceeded at the 
                         smallest allowed time step.
                     0 : simulation ended without meeting end condition (it hit
                         max_time).
                    >0 : simulation reached end condition successfully. Return
                         the result of end_condition.
    """

    y_cur = np.array(initial_guess)

    times = []
    ys = []
    current_time = start_time
    cur_step_size = base_time_step
    next_relax = -1 #if >0 this is the next time to increase the time step

    end_cond_val = True #set to false if the end condition is met 

    status = 0 #status of the solver
    not_failed = True #set to false if the solver fails

    while current_time < max_time and end_cond_val and not_failed:
        end_val = end_condition(current_time, y_cur)
        if  end_val != 0:
            end_cond_val = False #we hit the end condition
            status = abs(end_val) #success!
        else:
            #first check if the time step should relax
            if cur_step_size < base_time_step and current_time >= next_relax:
                #the step should relax, double it
                next_relax = -1
                cur_step_size = cur_step_size*2
                if cur_step_size > base_time_step:
                    #make sure the base time step isn't exceeded
                    cur_step_size = base_time_step

            #get the current value of the param to monitor
            monitor_cur = y_cur[param_to_monitor]

            #run the function to get the new derivative values
            deltas = np.array(func(cur_step_size, y_cur))

            #calculate the new values
            y_new = y_cur + deltas*cur_step_size

            #get the new monitor parameter
            monitor_new = y_new[param_to_monitor]

            #check the percent change in the monitor
            if abs(monitor_new - monitor_cur)/monitor_new > max_param_delta:
                #the change was larger than allowed. Reduce the step size and
                #try again

                #first check if the step size is already minimal, fail if so
                if cur_step_size == min_step_time:
                    not_failed = False
                    status = -1 #time step fail
                else:
                    #not at the smallest allowed step, so halve the step size
                    cur_step_size = cur_step_size/2
                    if cur_step_size < min_step_time:
                        cur_step_size = min_step_time

                    if next_relax < 0:
                        #the time to try increasing step size isn't set, set it
                        next_relax = current_time + base_time_step
                
            else:
                #the step succeeded, add the new values and increment
                current_time += cur_step_size
                ys.append(y_new)
                times.append(current_time)
                y_cur = y_new

    return times, ys, status



def simulate_particle(input_mass, input_vel, input_theta, co2_percent=-1):
    """
    Top level function to simulate a micrometeorite using the dynamic ODE solver
    in this script.

    Inputs:
        input_mass    - the initial mass of the micrometeorite [kg]
        input_vel     - the initial entry velocity of the micrometeorite [m s-1]
        input_theta   - initial entry angle of the micrometeorite [radians]
        co2_percent   - CO2 mass fraction. If set to -1 use O2, not CO2. NOTE:
                        this is actually a fraction between 0 and 1, not a 
                        percent.

    Returns:
        res - the output from dynamic_ode_solver() with the max temperature
              appended to the tuple
    """


    def should_end(_, y_cur, tracker):
        """
        Stop the solver when the particle has solidified after melting. Also
        stop the calculation if the particle is smaller than our minimum radius
        of 1 [micron] (we'll throw at all micrometeorites below 2 [microns] 
        later). We also stop if the particle is moving away from the Earth due 
        to a shallow entry angle.

        Inputs:
            _       - placeholder for time
            y_cur   - current simulation values
            tracker - object that stores whether to stop or not.

        Returns:
            0 - don't stop the simulation
            1 - stop because radius is too small
            2 - stop because the micrometeorite has solidified
            3 - stop because the impact angle was too shallow (moving away from
                Earth)
        """

        result = 0
        rad = get_radius_and_density(y_cur[3], y_cur[4])[0]

        if rad < 1E-6:
            #less than 1 micron radius
            result = 1
        if tracker["solidified"]:
            result = 2
        elif tracker["d_alt_dt"] > 0:
            #altitude is growing
            result = 3
        return result


    def sim_func(time, y_in, tracker):
        """
        The callable function passed to dynamic_ode_solver()
        
        Inputs:
            time - time at which to calculate [s]
            y_in - inputs to the function, has the form:
                   [initial altitude [km],
                    initial impact angle [radians],
                    initial Fe mass [kg],
                    initial FeO mass (always 0) [kg],
                    initial temperature [K]
                   ]
            tracker - holds the previous time to find the time step

        Returns:
            dy_dt - the derivative of each y value at t
        """
        co2_surface_vol = co2_percent/44.01/(co2_percent/44.01+(1-co2_percent)/28.014)*100
        # convert mass ratio to mixing ratio
        co2_series = np.array([1,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95])
        for i in range(len(co2_series) - 1):
            co2_lower, co2_upper = co2_series[i], co2_series[i+1]
            if co2_lower <= co2_surface_vol <= co2_upper:
                break
            
        #with open(f'./logfile/Redo_Output_CO2_{int(co2_surface_vol)}_oxi_extrp.log','a') as f:
        #    with redirect_stdout(f):
        alt, vel, theta, mass_fe, mass_feo, temp = y_in

        time_step = time 

        #avoiding breaking the particle is completely evaporated
        if mass_fe < 0:
            mass_fe = 0
        if mass_feo < 0:
            mass_feo = 0

        if mass_fe == 0 and mass_feo == 0:
            return [0, 0, 0, 0, 0, 0]

        rad, rho_m = get_radius_and_density(mass_fe, mass_feo)

        #calculate gravity
        gravity = GRAV_CONST*EARTH_MASS/alt**2


        ########################Track Motion###########################

        #new radius from law of cosines
        new_alt = ((vel*time_step)**2 + alt**2 - \
                2*alt*vel*time_step*cos(theta))**0.5

        #alpha will always be acute since we use small time steps
        alpha = asin(vel*time_step*sin(theta)/new_alt)
        phi = pi - alpha - theta

        new_vel = (vel**2 + (gravity*time_step)**2 - \
                2*vel*gravity*time_step*cos(phi))**0.5

        new_theta = 0
        if phi > pi/2:
            #new theta is acute
            new_theta = asin(vel*sin(phi)/new_vel)
        else:
            #new theta is likely obtuse
            rho = asin(gravity*time_step*sin(phi)/new_vel)
            new_theta = pi - rho - phi


        d_alt_dt = (new_alt - alt)/time_step
        tracker["d_alt_dt"] = d_alt_dt
        d_theta_dt = (new_theta - theta)/time_step

        #######################End Motion###############################

        #calculate the atmospheric density and total oxygen density
        rho_a, o_wt, co2_wt = atmospheric_density_and_oxygen(alt,co2_surface_vol, co2_lower, co2_upper)
        #if co2_percent != -1:
            #CO2 is the oxidant, so store CO2 density in the oxidant variable
            #rho_o = rho_a*co2_percent #use rho_o to track CO2 density
        #calculate the velocity derivative
        d_vel_dt = (new_vel - vel)/time_step - 0.75*rho_a*vel**2/(rho_m*rad) 

        #vapor pressure for FeO [Pa] from equation 7
        if temp<0:
            print('p_v: wrong temp: ', temp)
        p_v = 10**(10.3-20126/temp)

        #a note about ox_enc. This variable is recording the total oxygen (in
        #kg s-1) that is being absorbed by the micrometeorite. For CO2 reactions
        #this calculation uses kinetics while reactions with O2 just follow the
        #total oxygen concentration. See equation 10 for details. 
        ox_enc = 0
        ox_test_enc = 0 #to account for O2 in CO2-N2_O2 atmosphere
        rate_co2,add_o,add_co2,co2_rho,ram_p = 0,0,0,0,0
        if temp > temp_thre:
            if co2_percent != -1:
                #this is oxidation via CO2, use kinetics
                co2_rho = rho_a*co2_wt# [kg/m3] 
                o_rho = rho_a*o_wt# [kg/m3]                     
                S = pi*rad**2 #m2
                ram_p = 1/2*vel**2*co2_rho #[kg m-1 s-2]
                k_rate, _ = fe_co2_rate_pressure(temp, ram_p) #mol m-2 s-1 Pa-1 # Scenario 2
                #k_rate = fe_co2_rate_constant(temp)  #[mol m-2 s-1 Pa-1] # Scenario 3
                add_co2 = vel*M_O*S*co2_rho/M_CO2 #[kg s-1]
                Gamma_d = 1.3 # Drag coefficient in Deluca+ 2019
                Cd = 2*Gamma_d # Drag coefficient commonly used
                rate_co2 = Cd * S * M_O * k_rate * ram_p#[kg s-1]
                add_o = vel*S*o_rho #[kg s-1]
                
                if rate_co2<add_co2:
                    ox_enc = rate_co2 + add_o
                else:
                    ox_enc = add_co2 + add_o #Scenario 1
                
                #ox_enc = add_co2 + add_o
            else:
                #let oxygen be absorbed following equation 10.
                rho_o2 = 0
                ox_enc = GAMMA*rho_o2*pi*rad**2*vel
                

            #if ADD_OX_EST is set to true, add O2 to the model run 
            if ADD_OX_EST:
                o2_vol_frac = 0.01 #the 1% O2 volume % (fraction)
                #we want 1% O2 by volume, so we'll have to convert to wt %
                #first get the CO2 from wt% to vol % (but as a fraction)
                co2_vol_frac = 7*co2_percent/(11-4*co2_percent)

                #get the N2 vol % (fraction), remove the O2 as well
                n2_vol_frac = 1.0 - co2_vol_frac - o2_vol_frac
                o2_wt_perc = o2_vol_frac*M_O*2/(o2_vol_frac*M_O*2 + 
                        co2_vol_frac*M_CO2 + n2_vol_frac*M_N2)
                ox_test_enc = o2_wt_perc*rho_a*pi*rad**2*vel
            
        #the Langmuir formula for mass loss rate in [kg s-1] of FeO (eq. 6)
        dm_evap_fe_dt = 0 #if we need to evaporate Fe store it here
        if M_FEO/(2*pi*GAS_CONST*temp)<0:
            print('wrong temperature: ', temp)
        
        dm_evap_dt = 4*pi*rad**2*p_v*sqrt(M_FEO/(2*pi*GAS_CONST*temp)) #FeO evap

        if dm_evap_dt*time_step > mass_feo: 
            #we're evaporating more FeO than exists, so evaporate Fe as well and
            #find what fraction of dt we evaporate FeO, then the rest of dt
            #we'll assume Fe is evaporating
            feo_evap_frac = mass_feo/dm_evap_dt/time_step #FeO evaporate frac
            fe_evap_frac = 1.0 - feo_evap_frac
            p_v_fe = 10**(11.51 - 1.963e4/temp) #Fe evap rate (eq. 8)
            dm_evap_fe_dt = 4*pi*rad**2*p_v_fe*sqrt(M_FE/(2*pi*GAS_CONST*temp))
            dm_evap_fe_dt *= fe_evap_frac
            dm_evap_dt *= feo_evap_frac

        dmass_feo_dt = -dm_evap_dt + (M_FEO/M_O)*(ox_enc + ox_test_enc)
        dmass_fe_dt = -(M_FE/M_O)*(ox_enc + ox_test_enc) - dm_evap_fe_dt
        
        #print(f"{vel:.6g}, {temp:.6g}, {(alt-EARTH_RAD)/1000:.6g}, {ox_enc:.3g}, {rate_co2:.3g}, {add_o: .3g}, {add_co2:.3g}, {ram_p:.6g}")
    
        #combine all the evaporative loses here
        #NOTE: the latent heat of FeO=Fe for evaporation in our model.
        total_evap_dt = dm_evap_fe_dt + dm_evap_dt
        
        #oxidation via CO2 is endothermic so DELTA_H_OX is negative
        DELTA_H_OX = DELTA_H_OX_CO2
        if co2_percent == -1:
            #oxidation via oxygen is exothermic
            DELTA_H_OX = DELTA_H_OX_O2

        ox_test_qt_ox_dt = 0
        if ADD_OX_EST:
            #we need to account for the oxidation energy from O2 as well
            ox_test_qt_ox_dt = DELTA_H_OX_O2*(M_FEO/M_O)*ox_test_enc

        #total heat of oxidation
        dq_ox_dt = DELTA_H_OX*(M_FEO/M_O)*ox_enc + ox_test_qt_ox_dt 

        #the change in temperature (eq. 5)
        dtemp_dt = 1/(rad*C_SP*rho_m)*\
                    (3*rho_a*vel**3/8 - 3*L_V*total_evap_dt/(4*pi*rad**2) - 
                    3*SIGMA*temp**4 + 3*dq_ox_dt/(4*pi*rad**2))
        
        if temp<0:
            print(dtemp_dt)

        #check the temperatures, stop the simulation if the temp has peaked
        #and solidified
        #first set the peak temperature if needed
        if temp > tracker["peak_temp"]:
            tracker["peak_temp"] = temp

        if temp < tracker["peak_temp"]/2.5 and temp < temp_thre: #FEO_MELTING_TEMP:
            
            #one last check on the temperature. Sometimes the solver oscillates
            #for a step or two and can trigger an end, only end if the step was
            #less than a 10% change in temp. The oscillation will be caught as
            #an error and trigger a rerun with a smaller step size.
            if temp > tracker["last_temp"]*0.85:
                #setting this to True stops the simulation
                tracker["solidified"] = True
            if temp<0:
                print(temp)
        tracker["last_temp"] = temp


        return [d_alt_dt, 
                d_vel_dt, 
                d_theta_dt,
                dmass_fe_dt,
                dmass_feo_dt,
                dtemp_dt]

    #collect the initial values for dynamic_ode_solver()
    y_0 = [190000+EARTH_RAD, #initial altitude, 190 [km]
           input_vel, #initial velocity [m s-1]
           input_theta, #initial impact angle [radians]
           input_mass, #initial Fe mass [kg]
           0, #initial FeO mass [kg], always 0 at start
           300] #initial temperature of micrometeorite [K], not important

    #we need delta_t and states, so track the time and melt state with this obj
    tracker = {"time": 0, "solidified": False, "peak_temp": 0, "last_temp":0,
            "d_alt_dt":0} 

    end_cond = lambda t, y: should_end(t, y, tracker)

    start_time = 0 #[s]
    max_time = 300 #[s] usually only need 5-20 seconds, return error if hit 300
    param_to_monitor = 5 #monitor temperature
    max_param_delta = 0.001 #allow 0.1% change, max
    base_time_step = 0.01 #[s]
    min_step_time = 0.000001 #[s]
    
    res = dynamic_ode_solver(lambda t, y: sim_func(t, y, tracker), start_time, 
            max_time, y_0, param_to_monitor, max_param_delta, 
            base_time_step, min_step_time, end_cond)

    return res + (tracker["peak_temp"],)

def get_final_radius_and_fe_area_from_sim(data):
    """
    Calculate the final radius and the final Fe area fraction from the
    dynamic_ode_solver results.

    Inputs:
        data - the data from the dynamic_ode_solver() function

    Returns:
        rad     - final micrometeorite radius [m]
        fe_frac - final micrometeorite Fe fractional area
    """

    fe_mass = data[-1, 3]
    feo_mass = data[-1, 4]

    #replace negative values (we'll throw them out later anyway)
    if fe_mass < 0:
        fe_mass = 0
    if feo_mass < 0:
        feo_mass = 0

    rad = 0
    fe_frac = 0

    #make sure the micrometeorite didn't completely evaporate
    if fe_mass > 0 or feo_mass > 0:
        rad = get_radius_and_density(fe_mass, feo_mass)[0]
        fe_rad = get_radius_and_density(fe_mass, 0)[0]
        feo_rad = get_radius_and_density(0, feo_mass)[0]

        fe_area = pi*fe_rad**2
        feo_area = pi*feo_rad**2
        fe_frac = fe_area/(fe_area + feo_area)

    return [rad, fe_frac]

def readModelDataFile(filename):
    """
    Helper function to read model data saved to file.
    Read the data from an output file.

    Inputs:
        filename - the file to read

    Returns:
        result - the data from the file
    """
    
    file_obj = open(filename, "r")
    result = []
    data_not_started = True

    for line in file_obj:
        line_split = line.split()
        #make sure this isn't a label line
        if data_not_started:
            try:
                float(line_split[0])
                data_not_started = False
            except ValueError:
                continue

        if len(line_split) == 1:
            num_val = float(line_split[0])
            result.append(num_val)
        else:
            nums = []
            for num in line_split:
                num_val = float(num)
                nums.append(num_val)
            result.append(tuple(nums))

    return result



def saveModelData(data, filename, col_names=[]):
    """
    Helper function to write model output data to file.
    Takes an array and saves it to a file.

    Inputs:
        data      - input array to save
        filename  - the filename to use for the data
        col_names - the array of column names (must match len(data) to be used) 
    """

    file_obj = open(filename, "w")
    if len(col_names) == len(data[1]):
        line = ""
        for name in col_names:
            line += "%s    "%(name)
        line += "\n"
        file_obj.write(line)

    for d in data:
        line = ""
        if isinstance(d, (tuple, np.ndarray)):
            for item in d:
                line += "%2.10e "%item
        else:
            line += "%2.10e"%d
        line += "\n"
        file_obj.write(line)
    file_obj.close()


def multithreadWrapper(args):
    """
    This function will pass the multithreaded run arguments to simulateParticle
    then return the simulation parameters.

    Input:
        args - a tuple with the form (mass, velocity, impact angle, CO2 wt %)

    Returns:
        result - a tuple with the form (final radius [m], Fe fractional area, 
                 max temperature reached [K])
    """

    mass, velocity, theta, CO2_fac = args

    CO2_frac_wt = CO2_fac*44.01/(CO2_fac*44.01+(1-CO2_fac)*28.014)
    result = (0, 0)

    try:
        times, data, status, max_temp = simulate_particle(mass, velocity, theta, 
                co2_percent=CO2_frac_wt)
        final_radius, fe_area = get_final_radius_and_fe_area_from_sim(np.array(data))

        result = (final_radius, fe_area, max_temp)

        if status < 1:
            #the try failed, return the error value
            #NOTE: this is because the time step was too too large for the input
            #parameters. The only time this happens is for very fast, very large 
            #particles (that are very rare), so it has negligible impact.
            result = (-1, -1, -1)
            print("\nfailed run with:")
            print("status: %d"%(status))
            print("mass: %2.2e, vel: %0.1f [km s-1], theta: %0.1f, CO2(vol): %0.1f%%"%(
                mass, velocity/1000, theta*180/pi, CO2_fac*100
                ))
            print("------------------------------------------------")
    except:
        print("\nnumerical failure with:")
        print("mass: %2.2e, vel: %0.1f [km s-1], theta: %0.1f, CO2(vol): %0.1f%%"%(
            mass, velocity/1000, theta*180/pi, CO2_fac*100
            ))
        print("------------------------------------------------")
        result = (-1, -1, -1)

    return result


def generateRandomSampleData(num_samples=100, output_dir="rand_sim",
        input_dir=""):
    """
    Randomly sample from the input parameters (impact angle, velocity, radius)
    a given number of times and run the simulation. If an input directory is
    given the input data will be read from the file args_array.dat, in that 
    directory rather than with randomly generated inputs. 

    Inputs:
        num_samples - the number of simulations to run
        output_dir  - the directory to which the output file will be saved.
        input_dir   - the directory from which we should read inputs
    """

   #if __name__ == '__main__':

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    else:
        sys.stderr.write("The directory \""+output_dir+"\" already exists.\n")
        sys.stderr.write("Overwrite files in \""+output_dir+"\"? [y/n]: ")
        resp = 'y'#input()
        if resp not in ("y", "Y"):
            return

    #read from command line for CO2, if present
    CO2_fac = -1
    if len(sys.argv) == 2:
        CO2_fac = float(sys.argv[1])
    else:
        sys.stderr.write("The simulation is being run with O2.")

    thetas = np.zeros(num_samples)
    velocities = np.zeros(num_samples)
    masses = np.zeros(num_samples)
    if len(input_dir) > 0:
        #input directory given, read it
        sys.stderr.write("Reading data from: %s\n"%(input_dir))
        args = readModelDataFile(input_dir + "/args_array.dat")
        for i in range(len(args)):
            masses[i] = args[i][0]
            velocities[i] = args[i][1]
            thetas[i] = args[i][2]
    else:
        thetas = impactAngleDistribution().sample(size=num_samples)
        velocities = initialVelocityDistribution().sample(size=num_samples)
        velocities = velocities*1000 #convert from [km s-1] to [m s-1]
        masses = initialMassDistribution().sample(size=num_samples)
        masses = masses/1000 #convert from [g] to [kg]

    args_array = []
    for i in range(num_samples):
        args = (masses[i], velocities[i], thetas[i], CO2_fac)
        args_array.append(args)


    with Pool(cpu_count()-1) as p:
        results = list(tqdm(p.imap(multithreadWrapper, args_array), 
                            total=num_samples))

        saveModelData(args_array, output_dir+"/args_array.dat", 
                ["Mass [kg]    ", "Velocity [m s-1]", "Theta [rad]", 
                    "CO2 [ frac]"])
        saveModelData(results, output_dir+"/results.dat", ["Radius [m]   ",
            "Fe Area [frac]", "Max Temp [k]"])

        #delete runs with -1 in them, these failed to converge
        #save the cleaned up versions
        results = np.array(results)
        args_array = np.array(args_array)
        bad_val_inds = np.argwhere(results < 0)
        results = np.delete(results, bad_val_inds[:, 0], 0)
        args_array = np.delete(args_array, bad_val_inds[:, 0], 0)
        saveModelData(args_array, output_dir+"/clean_args_array.dat",
                ["Mass [kg]    ", "Velocity [m s-1]", "Theta [rad]", 
                    "CO2 [frac]"])
        saveModelData(results, output_dir+"/clean_results.dat", 
                ["Radius [m]   ", "Fe Area [frac]", "Max Temp [k]"])



def plot_particle_parameters(input_mass, input_vel, input_theta, CO2_frac,
        max_step=0.005):
    """
    Function to generate Figure 1. This will plot the various parameters of the
    simulation. 

    Inputs:
        input_mass  - the initial micrometeorite mass [kg]
        input_vel   - initial micrometeorite velocity [m s-1]
        input_theta - initial impact angle at top of atmosphere [radians]
        CO2_frac     - mixing ratio of the atmosphere that is CO2 [dimensionless]
        max_step    - maximum timestep to use in the simulation [s]

    Returns:
        no return, but generates a plot
    """
    CO2_frac_wt = CO2_frac*44.01/(CO2_frac*44.01+(1-CO2_frac)*28.014) # convert mixing ratio to mass ratio
    print('The CO2 wt frac is:', CO2_frac_wt) 
    times, data, stat, max_temp = simulate_particle(input_mass, input_vel, 
            input_theta, co2_percent=CO2_frac_wt)

    #print("status: %d"%(stat))

    times = np.array(times)
    data = np.array(data)

    alts = data[:, 0]
    velocities = (data[:, 1]**2 + data[:, 2]**2)**0.5
    fe_fracs = data[:, 3]/(data[:, 3] + data[:, 4])
    rads = get_radius_and_density(data[:, 3], data[:, 4], not_array=False)[0]
    temps = data[:, 5]

    start_ind = -1
    end_ind = -1
    last_ind = -1
    #track when the micrometeorite is molten so we can color the curves
    for i in range(0, len(temps)):
        if start_ind < 0 and temps[i] > FE_MELTING_TEMP:
            start_ind = i
        if end_ind < 0 and start_ind > 0 and temps[i] < FE_MELTING_TEMP:
            end_ind = i-1
        if temps[i] > FEO_MELTING_TEMP:
            last_ind = i

    print("Molten start: %0.1f seconds"%(times[start_ind]))
    print("Molten end: %0.1f seconds"%(times[end_ind]))


    rad, frac = get_final_radius_and_fe_area_from_sim(data)
    mass_frac = fe_fracs[-1]
    
    print("Starting radius: %0.1f [microns]"%(rads[0]*(1.0E6)))
    print("Final radius: %0.1f [microns]"%(rad/(1.0E-6)))
    print("Final Fe area fraction: %0.2f"%(frac))
    print("Final Fe mass fraction: %0.2f"%(mass_frac))
    ind = np.argmax(temps)
    print("Max temp: %0.0f [K]"%(temps[ind]))
    print("Altitude of max temp: %0.1f [km]"%((alts[ind]-EARTH_RAD)/1000))
    
    final_radius = rad/(1.0E-6)
    final_iron_area_frac = frac
    final_iron_mass_frac = mass_frac
    ind = np.argmax(temps)
    max_temp = temps[ind]
    altitude = (alts[ind]-EARTH_RAD)/1000
   
    #set the font size
    melt_start_ind = -1
    melt_end_ind = -1
    start_1000_ind = -1
    end_1000_ind = -1
    for i in range(len(times)):
        if melt_start_ind<0 and temps[i]>FEO_MELTING_TEMP:
            melt_start_ind = i
        if melt_start_ind>0 and melt_end_ind<0 and temps[i]<FEO_MELTING_TEMP:
            melt_end_ind = i
        if start_1000_ind<0 and temps[i] >1000:
            start_1000_ind = i
        if start_1000_ind>0 and end_1000_ind<0 and melt_end_ind > 0 and temps[i]<1000:
            end_1000_ind = i
    plot_this_figure = False

    if plot_this_figure:
        plt.rcParams.update({'font.size': 14})
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(4.5, 4), sharex=True)  

        ax1.plot(times, temps, ls = ':')
        ax1.plot(times[start_1000_ind:end_1000_ind], temps[start_1000_ind:end_1000_ind], color='green')
        ax1.plot(times[melt_start_ind:melt_end_ind], temps[melt_start_ind:melt_end_ind], color='yellow')
        ax1.plot(times[start_ind:end_ind], temps[start_ind:end_ind], color='red')
        ax1.set_ylabel("Temp. [K]")
        ax1.text(0.025, 0.9, "A", horizontalalignment='center',verticalalignment='center',transform=ax1.transAxes)

        ax2.plot(times, fe_fracs,ls = ':')
        ax2.plot(times[start_1000_ind:end_1000_ind], fe_fracs[start_1000_ind:end_1000_ind], color='green')
        ax2.plot(times[melt_start_ind:melt_end_ind], fe_fracs[melt_start_ind:melt_end_ind], color='yellow')
        ax2.plot(times[start_ind:end_ind], fe_fracs[start_ind:end_ind], color='red')
        ax2.set_ylabel("Fe Frac.")
        ax2.text(0.025, 0.9, "B", horizontalalignment='center',verticalalignment='center',transform=ax2.transAxes)
        fig.tight_layout()
        plt.show()
    
        plt.rcParams.update({'font.size': 14})
        fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, figsize=(4.5, 8), sharex=True)

        ax1.plot(times, temps)
        ax1.plot(times[start_ind:end_ind], temps[start_ind:end_ind], color="#ff7f0e")
        if last_ind - end_ind > 0:
            ax1.plot(times[end_ind:last_ind], temps[end_ind:last_ind], color="red")
        ax1.set_ylabel("Temp. [K]")
        ax1.text(0.025, 0.9, "A", horizontalalignment='center',verticalalignment='center',transform=ax1.transAxes)
        
        ax2.plot(times, velocities/1000)
        ax2.plot(times[start_ind:end_ind], velocities[start_ind:end_ind]/1000,color="#ff7f0e")
        if last_ind - end_ind > 0:
            ax2.plot(times[end_ind:last_ind], velocities[end_ind:last_ind]/1000,color="red")
        ax2.set_ylabel(r"Vel. [km s$^{-1}$]")
        ax2.text(0.025, 0.9, "B", horizontalalignment='center',verticalalignment='center',transform=ax2.transAxes)

        rads = np.array(rads)*(1.0E6)
        ax3.plot(times, rads)
        ax3.plot(times[start_ind:end_ind], rads[start_ind:end_ind], color="#ff7f0e")
        if last_ind - end_ind > 0:
            ax3.plot(times[end_ind:last_ind], rads[end_ind:last_ind], color="red")
        ax3.set_ylabel(r"Rad. [$\mu$m]")
        ax3.text(0.025, 0.9, "C", horizontalalignment='center',verticalalignment='center',transform=ax3.transAxes)

        ax4.plot(times, fe_fracs)
        ax4.plot(times[start_ind:end_ind], fe_fracs[start_ind:end_ind],color="#ff7f0e")
        if last_ind - end_ind > 0:
            ax4.plot(times[end_ind:last_ind], fe_fracs[end_ind:last_ind], color="red")
        ax4.set_ylabel("Fe Frac.")
        ax4.text(0.025, 0.9, "D", horizontalalignment='center',verticalalignment='center',transform=ax4.transAxes)

        alts = (alts-EARTH_RAD)/1000
        ax5.plot(times, alts)
        ax5.plot(times[start_ind:end_ind], alts[start_ind:end_ind], 
                color="#ff7f0e")
        if last_ind - end_ind > 0:
            ax5.plot(times[end_ind:last_ind], alts[end_ind:last_ind], 
                    color="red")
        ax5.set_ylabel("Alt. [km]")
        ax5.set_xlabel("Time [s]")
        ax5.text(0.025, 0.9, "E", horizontalalignment='center',verticalalignment='center',transform=ax5.transAxes)

        fig.tight_layout()
        fig.subplots_adjust(hspace=0.15)
        plt.show()
    
    return final_radius, final_iron_area_frac,final_iron_mass_frac,max_temp,altitude


##############################PLOTTING FUNCTIONS################################
#NOTE: if you don't download the generated data from the supplemental material
#      you must generate your own data before using the plotting functions for
#      figures 2, 3, and 4.
if __name__ =='__main__':
    #FIGURE 11
    #NOTE: for pure Fe, 50 micron radius has mass 3.665E-9 kg
    #50microns
    #res = plot_particle_parameters(3.665E-9, 13000, 45*pi/180, CO2_frac=0.1) #CO2_frac can be changed to 0.5
    ###########################DATA GENERATION FUNCTIONS############################

    #simulate micrometeorites entering the modern atmosphere
    # FIGURE 9 and S3
    if len(sys.argv) != 2:
        print("Usage: python script_name.py <CO2_wt_frac>")
        sys.exit(1)

    co2_frac = float(sys.argv[1])
    output_dir = "./Scenario_case/case_name/co2_%0.0f" % (co2_frac * 100)

    generateRandomSampleData(num_samples=200, output_dir=output_dir, input_dir="")


