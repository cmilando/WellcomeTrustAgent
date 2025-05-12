#
from config import exposure_profiles, time_activity_dict
import random
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import matplotlib.pyplot as plt
import pprint as pp


# /////////////////////////////////////////////////////////////////////////////
class Person(object):
    """"
    For each person, define an id, whether they have AC, and a time-activity str
    """

    def __init__(self,
                 person_id: int,
                 profile: str,
                 home_has_ac: bool,
                 time_activity_str: str):

        # set id
        self.person_id = person_id

        # do they have AC
        self.home_has_ac = home_has_ac

        #
        self.profile = profile

        # set time-activity profile
        self.time_activity_str = time_activity_str
        self.time_activity = time_activity_dict[time_activity_str]

        # set an empty vector for temperature exposure
        self.temperature_exposure = [None] * 24

    # print method
    def __str__(self):
        return 'Person_' + str(self.person_id)

    # So that it looks nice in lists
    __repr__ = __str__

    # get daily temperature exposure
    def get_daily_exposure(self):

        # initialize
        current_hour = int(0)
        current_zone_hour_str = str(current_hour)

        # get zone and AC
        current_zone = self.time_activity.get(current_zone_hour_str)

        if current_zone == 'Home':
            if self.home_has_ac:
                ac_str = 'has_ac'
            else:
                ac_str = 'no_ac'
            exp_prof_dict = exposure_profiles[current_zone].get(ac_str)
        else:
            this_key = list(exposure_profiles[current_zone])
            assert len(this_key) == 1
            exp_prof_dict = exposure_profiles[current_zone].get(this_key[0])

        # get the current mean and sd
        current_temp_mean = exp_prof_dict.get(current_zone_hour_str)[0]
        current_temp_sd = exp_prof_dict.get(current_zone_hour_str)[1]

        #
        self.temperature_exposure[current_hour] = (
            round(random.gauss(mu=current_temp_mean,
                               sigma=current_temp_sd), 3))

        # print(self.time_activity)

        for hour_i in range(1, 24):

            this_hour = str(hour_i)
            # first get new zone
            if self.time_activity.get(this_hour):
                current_zone_hour_str = str(hour_i)
            else:
                current_zone_hour_str = current_zone_hour_str

            # get zone and AC
            current_zone = self.time_activity.get(current_zone_hour_str)
            # print (new_hour + ' - ' + current_hour_str + ' - ' + current_zone)

            if current_zone == 'Home':
                if self.home_has_ac:
                    ac_str = 'has_ac'
                else:
                    ac_str = 'no_ac'
                exp_prof_dict = exposure_profiles[current_zone].get(ac_str)
            else:
                this_key = list(exposure_profiles[current_zone])
                assert len(this_key) == 1
                exp_prof_dict = exposure_profiles[current_zone].get(this_key[0])

            # get the current mean and sd
            current_temp_mean = exp_prof_dict.get(this_hour)[0]
            current_temp_sd = exp_prof_dict.get(this_hour)[1]

            #
            self.temperature_exposure[hour_i] = round(random.gauss(mu=current_temp_mean,
                                                                   sigma=current_temp_sd), 2)


# /////////////////////////////////////////////////////////////////////////////
class Population(object):

    def __init__(self,
                 n_persons: int,
                 distribution_dict,
                 n_per_sim_group=200):  # Defaults to 200

        #
        self.n_persons = n_persons

        # get the distribution keys as a float
        self.dist_probs = [distribution_dict[s]['percentage'] for s in distribution_dict]
        self.dist_names = [s for s in distribution_dict]
        assert sum(self.dist_probs) == 1

        # so actually, just get standard error with N_SIM
        # now create your population
        self.Persons = dict()
        for i in range(0, n_per_sim_group * len(self.dist_names)):
            this_dist_choice = distribution_dict[self.dist_names[i % len(self.dist_names)]]
            person_id = 'Person' + str(i + 1)
            self.Persons[person_id] = (
                Person(person_id=i + 1,
                       profile=self.dist_names[i % len(self.dist_names)],
                       home_has_ac=this_dist_choice['home_has_ac'],
                       time_activity_str=this_dist_choice['time_activity_profile']))

    def __str__(self):
        # Count type frequencies
        type_counts = Counter(person.profile for person in self.Persons.values())

        # Format as string
        formatted = ", ".join(f"{t}: {c}" for t, c in type_counts.items())
        return formatted

    def get_population_exposure(self):
        # could update this to parallel
        # for key in self.Persons:
        #     self.Persons[key].get_daily_exposure()

        with ThreadPoolExecutor() as executor:
            list(executor.map(lambda p: p.get_daily_exposure(), self.Persons.values()))

    def summarize_population_exposure(self):
        # 1. Group temperature_exposures by type
        exposures_by_type = defaultdict(list)

        for person in self.Persons.values():
            exposures_by_type[person.profile].append(person.temperature_exposure)

        # 2. Calculate mean and standard error for each hour
        mean_by_type = {}
        se_by_type = {}

        for person_type, exposures in exposures_by_type.items():
            data = np.array(exposures)  # shape: (n_people, 24)
            mean_by_type[person_type] = data.mean(axis=0)
            se_by_type[person_type] = data.std(axis=0, ddof=1) / np.sqrt(data.shape[0])

        # 3. Plot
        hours = np.arange(24)
        plt.figure(figsize=(10, 6))

        for person_type in mean_by_type:
            mean = mean_by_type[person_type]
            se = se_by_type[person_type]
            plt.plot(hours, mean, label=f'Type {person_type}')
            plt.fill_between(hours, mean - se, mean + se, alpha=0.3)

        plt.xlabel("Hour of Day")
        plt.ylabel("Temperature Exposure")
        plt.title("Average Hourly Temperature Exposure by Type")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def calculate_total_outcomes(self):
        """
        Based on the population weights, calculate the overall intervention outcomes
        :return:
        """

        # get the distribution choices
        dist_choices = random.choices(self.dist_names, weights=self.dist_probs, k=self.n_persons)

        # 1. Group temperature_exposures by type
        exposures_by_type = defaultdict(list)

        for person in self.Persons.values():
            exposures_by_type[person.profile].append(person.temperature_exposure)

        # 2. Calculate mean and standard error for each hour
        mean_by_type = {}
        se_by_type = {}

        for person_type, exposures in exposures_by_type.items():
            data = np.array(exposures)  # shape: (n_people, 24)
            mean_by_type[person_type] = data.mean(axis=0)
            se_by_type[person_type] = data.std(axis=0, ddof=1) / np.sqrt(data.shape[0])

        # so now you have mean_by_type
        pp.pprint(mean_by_type)
