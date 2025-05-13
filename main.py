from models import Population
import random

# set the random seed so you can replicate results
random.seed(10)

# setting the population parameters up like a dictionary makes sure
# that each name will be unique
# These are the marginal distributions, so if you want to add an
# age group, add a new time-activity profile
distribution_dict = {
    'Std_no_AC': {
        'percentage': 0.25,
        'time_activity_profile': 'Standard - Weekday',
        'home_has_ac': False
    },
    'Std_w_AC': {
        'percentage': 0.25,
        'time_activity_profile': 'Standard - Weekday',
        'home_has_ac': True
    },
    'Std_w_AC_Tree': {
        'percentage': 0.25,
        'time_activity_profile': 'Standard w/ Trees - Weekday',
        'home_has_ac': True
    },
    'Outdoor_worker': {
        'percentage': 0.25,
        'time_activity_profile': 'Outdoor - Weekday',
        'home_has_ac': True
    },
}

# **********************************************************************
# NOTE -- top-down interventions will be implemented by changing the percentages
#         we can also program in some behavioral switches
#         (e.g., if temperature at home > X, then do Y)
# **********************************************************************

# initialize the population
testPop = Population(n_persons=200000,
                     distribution_dict=distribution_dict,
                     n_per_sim_group=200)
# print(testPop)

# then simulate a single day for this population
# I guess you could do uncertainty a couple ways
# (1) more people (10,000?)
# (2) a set population (2,000) but then do like 500 days and average
testPop.get_population_exposure()
# print(testPop.Persons['Person1'].temperature_exposure)
# print(testPop.Persons['Person2'].temperature_exposure)

# now summarize personal exposure in a graph,
# by distribution type and hour of day
testPop.summarize_population_exposure()

# now calculate total outcomes for this simulation
# this is the number of cooling degree hours in each population
testPop.calculate_total_outcomes(threshold=75)
