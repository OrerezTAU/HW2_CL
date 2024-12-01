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
        avg_empirical_errors, avg_true_errors, n_values = [], [], []
        for n in range(m_first,m_last,step):

            sum_empirical_error,sum_true_error  = 0,0

            for t in range(1, T):
                sum_empirical_error, sum_true_error = self.calc_sum_errors(k, n, sum_empirical_error, sum_true_error)

            avg_empirical_errors.append(sum_empirical_error / T)
            avg_true_errors.append(sum_true_error / T)
            n_values.append(n)

        avg_empirical_errors = np.array(avg_empirical_errors)
        avg_true_errors = np.array(avg_true_errors)
        n_values = np.array(n_values)

        self.plot_errors(n_values, avg_empirical_errors, avg_true_errors)


    def experiment_k_range_erm(self, m, k_first, k_last, step):
        """Finds the best hypothesis for k= 1,2,...,10.
        Plots the empirical and true errors as a function of k.
        Input: m - an integer, the size of the data sample.
               k_first - an integer, the maximum number of intervals in the first experiment.
               m_last - an integer, the maximum number of intervals in the last experiment.
               step - an integer, the difference between the size of k in each experiment.

        Returns: The best k value (an integer) according to the ERM algorithm.
        """
        for k in range(k_first,k_last,step):


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
        if interval1[0] > interval2[1] or interval2[0] > interval1[1]:
            return [0,0]
        return [max(interval1[0], interval2[0]), min(interval1[1], interval2[1])]


    def calc_intersection_err(self,interval1, interval2, err_prob):
        """
        This function calculates the intersection error of two intervals
        :param interval1: The first interval
        :param interval2: The second interval
        :param err_prob: The probability of error in the intersection
        :param err: The current error
        :return: The updated error
        """
        intersection = self.intersect(interval1, interval2)
        if intersection[0] == 0 and intersection[1] == 0:
            return 0
        elif intersection[0] < intersection[1]:
            err = err_prob * (intersection[1] - intersection[0])
            return err
        return 0

    def calc_true_error(self,interval_list):
        """
        This function calculates the true error of a given interval list
        :param interval_list: The list of intervals
        :return: The true error of the interval list
        """
        likely_1_intervals = [[0, 0.2], [0.4, 0.6], [0.8, 1]]
        likely_0_intervals = [[0.2, 0.4], [0.6, 0.8]]
        err_prob_on_likely_1 = 0.2
        err_prob_on_likely_0 = 0.1
        err_likely_0 ,  err_likely_1 = [], []
        for interval in interval_list:
            for likely_1_interval in likely_1_intervals:
                err_num = self.calc_intersection_err(interval, likely_1_interval, err_prob_on_likely_1)
                err_likely_1.append(err_num)
                # for each interval in the list, calculate the error on intersections
                # with intervals where the probability of 1 is high

            for likely_0_interval in likely_0_intervals:
                err_num = self.calc_intersection_err(interval, likely_0_interval, err_prob_on_likely_0)
                err_likely_0.append(err_num)
                # for each interval in the list, calculate the error on intersections
                # with intervals where the probability of 0 is high

        return np.sum(err_likely_1) + np.sum(err_likely_0)

    def plot_errors(self, n_values, avg_empirical_errors, avg_true_errors):
        """
        This function plots the empirical and true errors
        :param n_values: The values of n
        :param avg_empirical_errors: The average empirical errors
        :param avg_true_errors: The average true errors
        """
        plt.clf()  # Clear current figure
        plt.close()
        plt.figure(figsize=(10, 6))
        plt.plot(n_values, avg_empirical_errors, marker='o',linestyle='-', color='b',label="Empirical Error")
        plt.plot(n_values, avg_true_errors, marker='s', linestyle='-', color='r',label="True Error")
        plt.title('Empirical Error vs. True Error')
        plt.xlabel('n')
        plt.ylabel('Error')
        plt.legend()
        plt.grid()
        plt.show()


    def calc_sum_errors(self, k, n, sum_empirical_error, sum_true_error):
        """
        This function calculates the sum of empirical and true errors over n samples and k intervals
        :param k: the number of intervals
        :param n: the number of samples
        :param sum_empirical_error: the sum of empirical errors
        :param sum_true_error: the sum of true errors
        :return: the updated sum of empirical and true errors
        """
        data_samples = self.sample_from_D(n)
        sorted_indices = np.argsort(data_samples[:, 0])  # Indices that would sort the first column
        sorted_data = data_samples[sorted_indices] # Make second column sorted according to the first column
        xs = sorted_data[:, 0]
        ys = sorted_data[:, 1].astype(int)
        best_intervals, best_error = intervals.find_best_interval(xs, ys, k)
        expected_error = np.sum(best_error)/n
        true_error = self.calc_true_error(best_intervals)
        sum_empirical_error += expected_error
        sum_true_error += true_error
        return sum_empirical_error, sum_true_error

    #################################


if __name__ == '__main__':
    ass = Assignment2()
    #ass.experiment_m_range_erm(10, 100, 5, 3, 100)
    ass.experiment_k_range_erm(1500, 1, 10, 1)
    # ass.experiment_k_range_srm(1500, 1, 10, 1)
    # ass.cross_validation(1500)

