#################################
# Your name: Or Erez
#################################

import numpy as np
import matplotlib.pyplot as plt
import intervals


class Assignment2(object):
    """Assignment 2 skeleton.

    Please use these function signatures for this assignment and submit this file, together with the intervals.py.
    """

    def sample_from_D(self, m):
        """Sample m data samples from D.
        Input: m - an integer, the size of the data sample.

        Returns: np.ndarray of shape (m,2) :
                A two dimensional array of size m that contains the pairs where drawn from the distribution P.
        """
        data_samples = np.random.rand(m, 2)
        prob_good_interval = 0.8  # Probability of choosing 1 in the interval [0,0.2] U [0.4,0.6] U [0.8,1]
        prob_bad_interval = 0.1  # Probability of choosing 1 in the interval [0.2,0.4] U [0.6,0.8]
        for i in range(m):
            if data_samples[i][0] < 0.2 or (0.4 < data_samples[i][0] < 0.6) or data_samples[i][0] > 0.8:
                data_samples[i][1] = np.random.binomial(1, prob_good_interval)
            else:
                data_samples[i][1] = np.random.binomial(1, prob_bad_interval)
        return data_samples


    def experiment_m_range_erm(self, m_first, m_last, step, k, T):
        """Runs the ERM algorithm.
        Calculates the empirical error and the true error.
        Plots the average empirical and true errors.
        Input: m_first - an integer, the smallest size of the data sample in the range.
               m_last - an integer, the largest size of the data sample in the range.
               step - an integer, the difference between the size of m in each loop.
               k - an integer, the maximum number of intervals.
               T - an integer, the number of times the experiment is performed.

        Returns: np.ndarray of shape (n_steps,2).
            A two dimensional array that contains the average empirical error
            and the average true error for each m in the range accordingly.
        """
        # TODO: Implement the loop
        pass

    def experiment_k_range_erm(self, m, k_first, k_last, step):
        """Finds the best hypothesis for k= 1,2,...,10.
        Plots the empirical and true errors as a function of k.
        Input: m - an integer, the size of the data sample.
               k_first - an integer, the maximum number of intervals in the first experiment.
               m_last - an integer, the maximum number of intervals in the last experiment.
               step - an integer, the difference between the size of k in each experiment.

        Returns: The best k value (an integer) according to the ERM algorithm.
        """
        # TODO: Implement the loop
        pass

    def experiment_k_range_srm(self, m, k_first, k_last, step):
        """Run the experiment in (c).
        Plots additionally the penalty for the best ERM hypothesis.
        and the sum of penalty and empirical error.
        Input: m - an integer, the size of the data sample.
               k_first - an integer, the maximum number of intervals in the first experiment.
               m_last - an integer, the maximum number of intervals in the last experiment.
               step - an integer, the difference between the size of k in each experiment.

        Returns: The best k value (an integer) according to the SRM algorithm.
        """
        # TODO: Implement the loop
        pass

    def cross_validation(self, m):
        """Finds a k that gives a good test error.
        Input: m - an integer, the size of the data sample.

        Returns: The best k value (an integer) found by the cross validation algorithm.
        """
        # TODO: Implement me
        pass

    #################################
    # Place for additional methods

    def intersect(self,interval1, interval2):
        """
        This function receives two intervals and returns their intersection
        :param interval1: The first interval
        :param interval2: The second interval
        :return: The intersection of the two intervals
        """
        l = max(interval1[0], interval2[0])
        r = min(interval1[1], interval2[1])

        intersection = [l, r]
        return intersection

    def calc_intersection_err(self,interval1, interval2, err_prob, err):
        """
        This function calculates the intersection error of two intervals
        :param interval1: The first interval
        :param interval2: The second interval
        :param err_prob: The probability of error in the intersection
        :param err: The current error
        :return: The updated error
        """
        intersection = self.intersect(interval1, interval2)
        if intersection[0] < intersection[1]:
            err += err_prob * (intersection[1] - intersection[0])
            return err
        return 0

    def calc_e_p(self,interval_list):
        """
        This function calculates the empirical error of a given interval list
        :param interval_list: The list of intervals
        :return: The empirical error of the interval list
        """
        likely_1_intervals = [[0, 0.2], [0.4, 0.6], [0.8, 1]]
        likely_0_intervals = [[0.2, 0.4], [0.6, 0.8]]
        err_prob_on_likely_1 = 0.2
        err_prob_on_likely_0 = 0.9
        err = 0
        for interval in interval_list:
            for likely_1_interval in likely_1_intervals:
                err += self.calc_intersection_err(interval, likely_1_interval, err_prob_on_likely_1, err)
                # for each interval in the list, calculate the error on intersections
                # with intervals where the probability of 1 is high

            for likely_0_interval in likely_0_intervals:
                err += self.calc_intersection_err(interval, likely_0_interval, err_prob_on_likely_0, err)
                # for each interval in the list, calculate the error on intersections
                # with intervals where the probability of 0 is high

        return err

    #################################


if __name__ == '__main__':
    ass = Assignment2()
    ass.experiment_m_range_erm(10, 100, 5, 3, 100)
    ass.experiment_k_range_erm(1500, 1, 10, 1)
    ass.experiment_k_range_srm(1500, 1, 10, 1)
    ass.cross_validation(1500)

