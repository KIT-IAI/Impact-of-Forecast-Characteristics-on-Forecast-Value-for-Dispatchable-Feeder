#################################### Packages ####################################
import json
import pickle
import datetime as dt
from sqlite3 import Timestamp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.pyplot import figure
from casadi import *
import copy



#################################### Class House and Subclass Deterministic House ####################################
class House:
    def __init__(self, parameter, start_compute, end_compute):
        #Define results
        #Time of computation
        self.start_compute = start_compute
        self.end_compute = end_compute
        self.hours = pd.date_range(start=start_compute, end=end_compute, freq='H')
        # Use pandas unique instead of numpy to prevent sorting. Else it might cause errors later if days of the week
        # belong to two different months
        self.days = pd.unique(self.hours.day)
        #Time of schedule
        self.start_schedule = start_compute + dt.timedelta(hours=parameter['len_compute'])
        self.end_schedule = end_compute + dt.timedelta(hours=parameter['len_DS_extension'])
        self.index_DS = pd.date_range(start=self.start_schedule, end=self.end_schedule, freq='H')

    #Write decision variables into list
    def dec_var_to_list(self, stage):
        liste = []
        for var in self.dec_var(stage):
            liste.append(self.__dict__[var])
        return(liste)

    #Split list and write into decision variables
    def list_to_dec_var(self, stage, liste):
        it = iter(liste)
        for var in self.dec_var(stage):
            self.__dict__[var] = next(it)
    
    #Write optimal decision variables into results
    def dec_var_to_results(self, stage, index, length):
        for var in self.dec_var(stage):
            if var == 'e' and stage != 'stage1':
                self.__dict__[stage + '_' + var]['value'][index+1:index+1+length] = self.__dict__[var][1:1+length]
            else:
                self.__dict__[stage + '_' + var]['value'][index:index+length] = self.__dict__[var][0:length]
                
                
class DeterministicHouse(House):
    def __init__(self, parameter, start_compute, end_compute):
        #Define decision variables
        self.g = None
        self.g_plus = None 
        self.g_minus = None
        self.delta_g = None
        self.p = None
        self.p_plus = None
        self.p_minus = None
        self.e = None
        
        #Inheritance Superclass
        super().__init__(parameter, start_compute, end_compute)
        #Initialize results
        blank_shape = np.rec.fromarrays((self.index_DS, np.zeros(len(self.index_DS))), names=('time','value'))
        self.estimate_e = np.rec.fromarrays((self.days[1:], np.zeros(len(self.days[1:]))), names=('day','value'))
        self.stage1_forecast_power = blank_shape.copy()
        self.actual_power = blank_shape.copy()
        for stage in ['stage1', 'actual']:
            for var in self.dec_var(stage):
                self.__dict__[stage + '_' + var] =  blank_shape.copy()
        
    #Define decision variables for different stages
    def dec_var(self, stage):
        if stage == 'stage1':
            keys = ['g', 'g_plus', 'g_minus', 'p', 'p_plus', 'p_minus', 'e']
        elif stage == 'actual':
            keys = ['g', 'delta_g', 'p', 'p_plus', 'p_minus', 'e']
        else:
            print('Wrong stage description')
        return(keys)
    

#################################### Run All ####################################
def run_all_function(parameter, houses, forecast_prosumption, actual_prosumption,dict_to_save):
    
    ##Initialization for results
    forecast_median = dict.fromkeys(['house' + str(i) for i in range(len(houses))], None)
    actual_power = dict.fromkeys(['house' + str(i) for i in range(len(houses))], None)
    estimate_e = dict.fromkeys(['house' + str(i) for i in range(len(houses))], None)
    actual_e = dict.fromkeys(['house' + str(i) for i in range(len(houses))], None)
    DS = dict.fromkeys(['house' + str(i) for i in range(len(houses))], None)
    stage1_costs = []
    stage2_costs = []
    stage3_costs = []


    forecasts = []

    
    ########### Scheduling ########### 

    ####Day 0
    ###hour = 12:00

    ##Generate Forecasts for Median for intervall 12:00-13:00 upwards
    for i, house in enumerate(houses):
        forecast_median['house' + str(i)] = forecast_prosumption['house' + str(i)][0]

    ##Estimate SoC in 00:00 -> not needed in Day 0
    #Save results
    for i, house in enumerate(houses):
        actual_e['house' + str(i)] = house.e_start
        house.estimate_e['value'][0] = house.e_start
        house.actual_e['value'][0] = house.e_start 

    ##1. Stage: Compute DS; DS from 00:00 - 23:00 + extension 
    forecast_median_extract = {keys: values[parameter['len_compute']:] for keys, values in forecast_median.items()}
    stage1_costs.append(compute_DS(parameter, houses, forecast_median_extract, actual_e))
    #Save results
    for i, house in enumerate(houses):
        house.dec_var_to_results('stage1', 0, parameter['len_DS'])
        house.stage1_forecast_power['value'][0:parameter['len_DS']] = \
        forecast_median['house' + str(i)][parameter['len_compute']:parameter['len_compute']+parameter['len_DS']]

    forecasts.append(copy.deepcopy(forecast_median_extract['house0'][:30]))
    times = [houses[0].start_schedule + pd.Timedelta(days=x) for x in range(7)] 

    ####Day = 1,... 
    ###hour = 00:00,... 
    #index_time: index of actual time
    #time: actual time
    for index_time, time in enumerate(houses[0].hours[int(np.where(houses[0].hours == houses[0].start_schedule)[0]):], start=0):

        #index_forecast: index of actual time in forecast data
        index_forecast = index_time + parameter['len_compute']
        
        index_day = np.where(time.day == houses[0].days)[0]

        ##Generate Forecasts for Median from time upwards
        for i in range(len(houses)):
            forecast_median['house' + str(i)] = forecast_prosumption['house' + str(i)][index_day[0]]


        ########### Offline ########### 

        ##If 12:00: Estimate SoC and Compute DS
        if time.hour == 12 and time.day != houses[0].end_compute.day:

            #index_schedule: Start of schedule
            index_schedule = int(np.where((houses[0].index_DS.hour == 0) & \
                                          (houses[0].index_DS.day == (time + pd.Timedelta(days=1)).day))[0])

            ##Estimate SoC in 00:00
            #We need forecasts from 12:00 upwards
            #We need DS from 12:00 upwards
            #e is SoC in 12:00

            timestamp = time
            dict_per_timestamp = {}
            

            for i, house in enumerate(houses):
                DS['house' + str(i)] = house.stage1_g['value'][index_time:]
            estimate_e = estimate_SOE(parameter, houses, forecast_median, actual_e, DS)
            #Save results
            for i, house in enumerate(houses):
                house.estimate_e['value'][index_day] = estimate_e['house' + str(i)]
       
            
            ##1. Stage: Compute DS from 00:00 - 23:00
            #We need forecasts from 00:00 upwards 
            #e0 is SoC in 00:00
            forecast_median_extract = {keys: values[parameter['len_compute']:] for keys, values in forecast_median.items()}
            
            


            stage1_costs.append(compute_DS(parameter, houses, forecast_median_extract, estimate_e))
            #Save results
            for i, house in enumerate(houses):
                house.dec_var_to_results('stage1', index_schedule, parameter['len_DS_extended'])
                house.stage1_forecast_power['value'][index_schedule:index_schedule+parameter['len_DS']] = \
                forecast_median_extract['house' + str(i)][:parameter['len_DS']]




            forecasts.append(copy.deepcopy(forecast_median_extract['house0'][:30]))



                
        ########### Online ###########
        
        ##Get actual load power
        for i, house in enumerate(houses):
            actual_power['house' + str(i)] = [actual_prosumption['house' + str(i)][index_forecast]]
            DS['house' + str(i)] = house.stage1_g['value'][index_time]

        ##3. Stage: Controlled ESS
        stage3_costs.append(controlled_ESS(parameter, houses, actual_power, actual_e, DS))


        #Save results
        for i, house in enumerate(houses):
            house.dec_var_to_results('actual', index_time, 1)
            house.actual_power['value'][index_time] = actual_power['house' + str(i)][0]
            actual_e['house' + str(i)] = house.actual_e['value'][index_time+1]

    actual_g = [copy.deepcopy(houses[0].actual_g[24*x:24*(x+1)]['value']) for x in range(7)]
    stage1_g = [copy.deepcopy(houses[0].stage1_g[24*x:24*(x+1)]['value']) for x in range(7)]
    actual_prosumption_list = [copy.deepcopy(np.ravel(np.array(actual_prosumption['house0'][24*x+12:24*(x+1)+12]))) for x in range(7)]
    actual_e = [copy.deepcopy(houses[0].actual_e[24*x:24*(x+1)]['value']) for x in range(7)]
    stage1_g_plus = [copy.deepcopy(houses[0].stage1_g_plus[24*x:24*(x+1)]['value']) for x in range(7)]
    stage1_g_minus = [copy.deepcopy(houses[0].stage1_g_minus[24*x:24*(x+1)]['value']) for x in range(7)]
    stage1_e = [copy.deepcopy(houses[0].stage1_e[24*x:24*(x+1)]['value']) for x in range(7)]
    stage1_p = [copy.deepcopy(houses[0].stage1_p[24*x:24*(x+1)]['value']) for x in range(7)]


    estimate_SOE_per_timestamp = copy.deepcopy(houses[0].estimate_e)

    
    for i in range(7):
        tmp_dict = {}
        dict_helper_add_list(tmp_dict,"forecast",forecasts[i])
        dict_helper_add_list(tmp_dict,"actual_g",actual_g[i])
        dict_helper_add_list(tmp_dict,"stage1_g",stage1_g[i])
        dict_helper_add_list(tmp_dict,"stage1_g_plus",stage1_g_plus[i])
        dict_helper_add_list(tmp_dict,"stage1_g_minus",stage1_g_minus[i])
        dict_helper_add_list(tmp_dict,"gt",actual_prosumption_list[i])
        dict_helper_add_list(tmp_dict,"actual_e",actual_e[i])
        dict_helper_add_list(tmp_dict,"stage1_e",stage1_e[i])
        dict_helper_add_list(tmp_dict,"stage1_p",stage1_p[i])

        tmp_dict.update({"SoE":estimate_SOE_per_timestamp[i]['value']})

        tmp_time = times[i]

        dict_to_save.update({tmp_time:tmp_dict})
    

    print("done")   
    return stage3_costs, stage1_costs
 

#################################### 1 Stage OP ####################################
def compute_DS(parameter, houses, forecast_median_extract, estimate_e):
    #Define variables
    objective = 0
    #Initialize constraints
    constraints_list = []
    lb_list = []
    ub_list = []
    #Initialize list of all decision variables
    decision = []
    
    for house in houses:
        house.g = SX.sym('g', parameter['len_DS_extended'])
        house.g_plus = SX.sym('g+', parameter['len_DS_extended'])
        house.g_minus = SX.sym('g-', parameter['len_DS_extended'])
        house.p = SX.sym('p', parameter['len_DS_extended'])
        house.p_plus = SX.sym('p+', parameter['len_DS_extended'])
        house.p_minus = SX.sym('p-', parameter['len_DS_extended'])
        house.e = SX.sym('e', parameter['len_DS_extended']+1)
        #Extend list of all decision variables
        decision.extend(house.dec_var_to_list('stage1'))
    
        for k in range(0, parameter['len_DS_extended']):
            objective = objective + (parameter['costs']['single_plus']*house.g_plus[k] + \
                                     parameter['costs']['quadratic_plus']*house.g_plus[k]**2 + \
                                     parameter['costs']['single_minus']*house.g_minus[k] + \
                                     parameter['costs']['quadratic_minus']*house.g_minus[k]**2) 
                            
        
            #g(k) = g+(k) + g-(k)
            constraint_split_g = house.g[k] - house.g_plus[k] - house.g_minus[k]
            lb_split_g = 0
            ub_split_g = 0
            constraints_list.append(constraint_split_g)
            lb_list.append(lb_split_g)
            ub_list.append(ub_split_g)

            #g+(k) >= 0 
            constraint_g_plus = house.g_plus[k]
            lb_g_plus = 0
            ub_g_plus = parameter['g_max']
            constraints_list.append(constraint_g_plus)
            lb_list.append(lb_g_plus)
            ub_list.append(ub_g_plus)

            #g-(k) <= 0
            constraint_g_minus = house.g_minus[k]
            lb_g_minus = -parameter['g_max']
            ub_g_minus = 0
            constraints_list.append(constraint_g_minus)
            lb_list.append(lb_g_minus)
            ub_list.append(ub_g_minus)                 
        
    
    #(houses, constraints_list, lb_list, ub_list, length, forecast_median, e)
    define_constraints(houses, constraints_list, lb_list, ub_list, parameter['len_DS_extended'], forecast_median_extract,\
                       estimate_e)
    
    nlp = {}
    nlp['x'] = vertcat(*decision)
    nlp['f'] = objective
    nlp['g'] = np.asarray(constraints_list)
    lower_bound = np.asarray(lb_list)
    upper_bound = np.asarray(ub_list)

    opts = {'ipopt.print_level':0, 'print_time':0}
    F = nlpsol('F','ipopt',nlp,opts)
    opti = F(x0 = np.zeros(vertcat(*decision).shape[0]), lbg = lower_bound, ubg = upper_bound)
    opti_decision_list = np.squeeze(opti['x'])
    
    it1 = iter(opti_decision_list) 
    sizes = [item.shape[0] for item in decision]
    it2 = iter([[next(it1) for _ in range(size)] for size in sizes])                 
    for house in houses:
        liste = [next(it2) for _ in range(len(house.dec_var('stage1')))]                
        house.list_to_dec_var('stage1', liste)
    
    assert F.stats()['return_status'] == 'Solve_Succeeded', \
    "Assertion in 1stage_ComputeDispatchSchedule:" + F.stats()['return_status']
    return(np.squeeze(opti['f']))


#################################### 3 Stage OP ####################################
def controlled_ESS(parameter, houses, actual_power, actual_e, DS):
    
    #Define variables
    objective = 0
    #Initialize constraints
    constraints_list = []
    lb_list = []
    ub_list = []
    #Initialize list of all decision variables
    decision = []
    
    for i, house in enumerate(houses):
        house.g = SX.sym('g', 1)
        house.delta_g = SX.sym('delta_g', 1)
        house.p = SX.sym('p', 1)
        house.p_plus = SX.sym('p+', 1)
        house.p_minus = SX.sym('p-', 1)
        house.e = SX.sym('e', 2)
        #Extend list of all decision variables
        decision.extend(house.dec_var_to_list('actual'))
    
        objective = objective + house.delta_g**2
    
        #g = g_ref + delta_g
        constraint_actual_g = house.g - DS['house' + str(i)] - house.delta_g
        lb_actual_g = 0
        ub_actual_g = 0
        constraints_list.append(constraint_actual_g)
        lb_list.append(lb_actual_g)
        ub_list.append(ub_actual_g)
    
    
    #(houses, constraints_list, lb_list, ub_list, length, forecast_median, e)
    define_constraints(houses, constraints_list, lb_list, ub_list, 1, actual_power, actual_e)
 

    nlp = {}
    nlp['x'] = vertcat(*decision)
    nlp['f'] = objective
    nlp['g'] = vertcat(*constraints_list)
    lower_bound = np.asarray(lb_list)
    upper_bound = np.asarray(ub_list)
    
    opts = {'ipopt.print_level':0, 'print_time':0}
    F = nlpsol('F','ipopt',nlp,opts)
    opti = F(x0 = np.zeros(vertcat(*decision).shape[0]), lbg = lower_bound, ubg = upper_bound)
    opti_decision_list = np.squeeze(opti['x'])
    
    it1 = iter(opti_decision_list) 
    sizes = [item.shape[0] for item in decision]
    it2 = iter([[next(it1) for _ in range(size)] for size in sizes])                 
    for house in houses:
        liste = [next(it2) for _ in range(len(house.dec_var('actual')))]                
        house.list_to_dec_var('actual', liste)  
    
    assert F.stats()['return_status'] == 'Solve_Succeeded', \
    "Assertion in 3stage_ControlledESS or EstimateSOC:" + F.stats()['return_status']
    return(np.squeeze(opti['f']))


#################################### Define Constraints ####################################
def define_constraints(houses, constraints_list, lb_list, ub_list, length, forecast_median, e):
   
    for k in range(0,length):  
        for i, house in enumerate(houses): 
            
            #e(k+1) = e(k) + delta*(p(k) - mu*p+(k) + mu*p-(k))
            constraint_SOC_balance = house.e[k+1] - house.e[k] - parameter['delta']*(house.p[k] - \
                                     parameter['mu']*house.p_plus[k] + parameter['mu']*house.p_minus[k])
            lb_SOC_balance = 0
            ub_SOC_balance = 0
            constraints_list.append(constraint_SOC_balance)
            lb_list.append(lb_SOC_balance)
            ub_list.append(ub_SOC_balance)

            #g(k) = p(k) + l(k) 
            constraint_power_exchange = house.g[k] - house.p[k] - forecast_median['house' + str(i)][k] 
            lb_power_exchange = 0
            ub_power_exchange = 0
            constraints_list.append(constraint_power_exchange)
            lb_list.append(lb_power_exchange)
            ub_list.append(ub_power_exchange)

            #p(k) = p+(k) + p-(k)
            constraint_split_power = house.p[k] - house.p_plus[k] - house.p_minus[k]
            lb_split_power = 0 
            ub_split_power = 0
            constraints_list.append(constraint_split_power)
            lb_list.append(lb_split_power)
            ub_list.append(ub_split_power)

            #p+(k) >= 0 
            constraint_p_plus = house.p_plus[k]
            lb_p_plus = 0
            ub_p_plus = house.p_max
            constraints_list.append(constraint_p_plus)
            lb_list.append(lb_p_plus)
            ub_list.append(ub_p_plus)

            #p-(k) <= 0 
            constraint_p_minus = house.p_minus[k]
            lb_p_minus = house.p_min
            ub_p_minus = 0
            constraints_list.append(constraint_p_minus)
            lb_list.append(lb_p_minus)
            ub_list.append(ub_p_minus)

            #p+(k)*p-(k) <= relaxation_p
            constraint_complement_power = house.p_plus[k]*(-house.p_minus[k])
            lb_complement_power = 0
            ub_complement_power = parameter['relaxation_p']
            constraints_list.append(constraint_complement_power)
            lb_list.append(lb_complement_power)
            ub_list.append(ub_complement_power)

            #e_min <= e(k+1) <= e_max
            constraint_SOC_bounds = house.e[k+1]
            lb_SOC_bounds = house.e_min
            ub_SOC_bounds = house.e_max
            constraints_list.append(constraint_SOC_bounds)
            lb_list.append(lb_SOC_bounds)
            ub_list.append(ub_SOC_bounds)
            
            #e(kb) = e
            if k == 0:
                constraint_e_start = house.e[k]
                lb_e_start = e['house' + str(i)]
                ub_e_start = e['house' + str(i)]
                constraints_list.append(constraint_e_start)
                lb_list.append(lb_e_start)
                ub_list.append(ub_e_start)

        #-g_max <= sum(g(k)) <= g_max
        constraint_restrict_sum_g = sum(house.g[k] for house in houses)
        lb_restrict_sum_g = -parameter['g_max']
        ub_restrict_sum_g = parameter['g_max']
        constraints_list.append(constraint_restrict_sum_g)
        lb_list.append(lb_restrict_sum_g)
        ub_list.append(ub_restrict_sum_g)



#################################### Estimate SOE ####################################
def estimate_SOE(parameter, houses, forecast_median, e, DS):
    e_temp = copy.deepcopy(e)
    for k2 in range(parameter['len_compute']):
        forecast_median_extract = {keys: values[k2:] for keys, values in forecast_median.items()}
        DS_extract = {keys: values[k2] for keys, values in DS.items()}
        #(parameter, houses, actual_power, actual_e, DS)
        controlled_ESS(parameter, houses, forecast_median_extract, e_temp, DS_extract)
        for i, house in enumerate(houses):
            e_temp['house' + str(i)] = house.e[1]
    return(e_temp)



#################################### Parameter ####################################
parameter = {'costs':{'single_plus': 0.3, 'quadratic_plus': 0.05, 'single_minus': 0.15, 'quadratic_minus': 0.05, \
             'alpha': 50, 'gamma': [0.05] + [0.1] + [0.2] + [0.4] + [1000000]*400},
             'e_start': 6.75, 
             #delta is distance between steps -> 1 hour
             'delta': 1,
             'mu': 0.05,
             'relaxation_p' : 1e-8}

parameter['len_DS'] = 24
#DS can be extended up to 36 hours
parameter['len_DS_extended'] = 30
parameter['len_DS_extension'] = parameter['len_DS_extended'] - parameter['len_DS']
parameter['len_compute'] = 12
parameter['len_cycle'] = parameter['len_compute'] + parameter['len_DS_extended']
parameter['number_houses'] = 1
parameter['g_max'] = 2000000
##Checks
#47 is length of energy quantile forecasts
assert(parameter['len_DS_extended'] - parameter['len_compute'] <= 47)


#################################### Cost functions ####################################

def costs_DS(house):
    return(sum(parameter['costs']['single_plus']*house.stage1_g_plus['value'][:-parameter['len_DS_extension']] + \
           parameter['costs']['quadratic_plus']*house.stage1_g_plus['value'][:-parameter['len_DS_extension']]**2 + \
           parameter['costs']['single_minus']*house.stage1_g_minus['value'][:-parameter['len_DS_extension']] + \
           parameter['costs']['quadratic_minus']*house.stage1_g_minus['value'][:-parameter['len_DS_extension']]**2))

def costs_imbalances2(house):
    return(sum(2*parameter['costs']['single_plus']*abs(house.stage1_g['value'][:-parameter['len_DS_extension']] - \
                                                       house.actual_g['value'][:-parameter['len_DS_extension']]) + \
           2*parameter['costs']['quadratic_plus']*(house.stage1_g['value'][:-parameter['len_DS_extension']] - \
                                                   house.actual_g['value'][:-parameter['len_DS_extension']])**2))

def costs_imbalances10(house):
    return(sum(10*parameter['costs']['single_plus']*abs(house.stage1_g['value'][:-parameter['len_DS_extension']] - \
                                                        house.actual_g['value'][:-parameter['len_DS_extension']]) + \
           10*parameter['costs']['quadratic_plus']*(house.stage1_g['value'][:-parameter['len_DS_extension']] - \
                                                    house.actual_g['value'][:-parameter['len_DS_extension']])**2))

def costs_rescheduling(house):
    return(sum((house.stage1_g['value'][:-parameter['len_DS_extension']] - \
                house.stage2_g['value'][:-parameter['len_DS_extension']])**2))


##################################

def dict_helper_add_list(dict, string , list):
    for idx, x in enumerate(list):
        tmp = string + "_" + str(idx)
        dict.update({tmp: x})
        
    return