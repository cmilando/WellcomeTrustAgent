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
        'percentage': 0.7,
        'time_activity_profile': 'Standard - Weekday',
        'home_has_ac': False
    },
    'Std_w_AC': {
        'percentage': 0.3,
        'time_activity_profile': 'Standard - Weekday',
        'home_has_ac': True
    }
}

# **********************************************************************
# NOTE -- top-down interventions will be implemented by changing the percentages
#         we can also program in some behavioral switches
#         (e.g., if temperature at home > X, then do Y)
# **********************************************************************

# initialize the population
testPop = Population(100, distribution_dict)
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
