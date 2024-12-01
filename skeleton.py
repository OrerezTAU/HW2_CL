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

        error_matrix = np.array([avg_empirical_errors, avg_true_errors])

        n_values = np.array(n_values)

        self.plot_errors(n_values, error_matrix[0], error_matrix[1],'n','Empirical Error vs. True Error')

        return error_matrix



    def experiment_k_range_erm(self, m, k_first, k_last, step):
        """Finds the best hypothesis for k= 1,2,...,10.
        Plots the empirical and true errors as a function of k.
        Input: m - an integer, the size of the data sample.
               k_first - an integer, the maximum number of intervals in the first experiment.
               m_last - an integer, the maximum number of intervals in the last experiment.
               step - an integer, the difference between the size of k in each experiment.

        Returns: The best k value (an integer) according to the ERM algorithm.
        """

        best_k, error_matrix, k_values = self.calc_errors_best_k(k_first, k_last, m, step)
        self.plot_errors(k_values,error_matrix[0],error_matrix[1],'k','Empirical Error vs. True Error')
        return best_k



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
        best_k, error_matrix, k_values = self.calc_errors_best_k(k_first, k_last, m, step)
        penalty_arr = np.array([2*np.sqrt((2*k + np.log(0.1 / k**2))/m) for k in k_values]) # Calculate the penalty
        index_min = np.argmin(penalty_arr + error_matrix[0])
        best_k = k_values[index_min]
        self.plot_with_penalty(error_matrix, k_values, penalty_arr)
        return best_k



    def cross_validation(self, m):
        """Finds a k that gives a good test error.
        Input: m - an integer, the size of the data sample.

        Returns: The best k value (an integer) found by the cross validation algorithm.
        """

        data = self.sample_from_D(m)
        data[1] = data[1].astype(int)

        # choose random indices for training and validation sets
        indices = np.random.permutation(len(data))

        # Split the indices
        train_size = int(0.8 * len(data))
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]

        # Create training and validation sets
        train_set, val_set = data[train_indices], data[val_indices]

        xs_train, ys_train = self.sort_data_samples(train_set)

        erm_list,empirical_errors = [], []

        # Find the best intervals for k=1,2,...,10
        for k in range(1, 11):
            best_intervals, best_error = intervals.find_best_interval(xs_train, ys_train, k)
            erm_list.append(best_intervals)
            empirical_errors.append(np.sum(best_error)/len(train_set))

        # Find the best intervals from the list of intervals regarding validation set
        best_intervals, min_error, best_index = self.find_val_best_intervals(erm_list, val_set)
        return best_index


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
        This function calculates the error for the intersection of two intervals
        :param interval1: The first interval
        :param interval2: The second interval
        :param err_prob: The probability of error in the intersection
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
        xs, ys = self.sort_data_samples(data_samples)
        best_intervals, best_error = intervals.find_best_interval(xs, ys, k)
        expected_error = np.sum(best_error)/n
        true_error = self.calc_true_error(best_intervals)
        sum_empirical_error += expected_error
        sum_true_error += true_error
        return sum_empirical_error, sum_true_error

    def sort_data_samples(self, data_samples):
        sorted_indices = np.argsort(data_samples[:, 0])  # Indices that would sort the first column
        sorted_data = data_samples[sorted_indices]  # Make second column sorted according to the first column
        xs = sorted_data[:, 0]
        ys = sorted_data[:, 1].astype(int)
        return xs, ys

    def calc_errors_best_k(self, k_first, k_last, m, step):
        """
        This function calculates the empirical and true errors for all k values in the range
        :param k_first: the first k value
        :param k_last: the last k value
        :param m: the number of samples
        :param step: the step size
        :return: the best k value, the error matrix and the k values
        """
        arr_empirical_error, arr_true_error, k_values = [], [], []
        min_empirical_error, best_k = 1, -1
        for k in range(k_first, k_last, step):
            sum_empirical_error, sum_true_error = self.calc_sum_errors(k, m, 0, 0)
            if sum_empirical_error < min_empirical_error:
                min_empirical_error = sum_empirical_error
                best_k = k
            arr_empirical_error.append(sum_empirical_error)
            arr_true_error.append(sum_true_error)
            k_values.append(k)
        error_matrix = np.array([arr_empirical_error, arr_true_error])
        k_values = np.array(k_values)
        return best_k, error_matrix, k_values

    def plot_errors(self, x_values, avg_empirical_errors, avg_true_errors, x_label, title):
        """
        This function plots the empirical and true errors
        :param title:
        :param x_label: the label of the x axis
        :param x_values: The values of n
        :param avg_empirical_errors: The average empirical errors
        :param avg_true_errors: The average true errors
        """

        plt.figure(figsize=(10, 6))
        plt.plot(x_values, avg_empirical_errors, marker='o', linestyle='-', color='b', label="Empirical Error")
        plt.plot(x_values, avg_true_errors, marker='o', linestyle='-', color='r', label="True Error")
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel('Error')
        plt.legend()
        plt.grid()
        plt.show()

    def plot_with_penalty(self, error_matrix, k_values, penalty_arr):
        """
        This function plots the empirical error, true error, penalty and empirical error + penalty
        :param error_matrix: the error matrix, containing the empirical and true errors
        :param k_values: the k values (x axis)
        :param penalty_arr: the penalty array
        """

        plt.figure(figsize=(10, 6))
        plt.plot(k_values, penalty_arr, marker='o', linestyle='-', color='g', label="Penalty")
        plt.plot(k_values, penalty_arr + error_matrix[0], marker='o', linestyle='-', color='y',
                 label="Empirical Error + Penalty")
        plt.plot(k_values, error_matrix[0], marker='o', linestyle='-', color='b', label="Empirical Error")
        plt.plot(k_values, error_matrix[1], marker='o', linestyle='-', color='r', label="True Error")
        plt.title('Empirical Error, True Error, Penalty and Empirical Error + Penalty')
        plt.xlabel('k')
        plt.ylabel('Error')
        plt.legend()
        plt.grid()
        plt.show()

    def calculate_validation_error(self, intervals_list, validation_data):
        """
        This function calculates the validation error of a given list of intervals
        :param intervals_list: the list of intervals
        :param validation_data: the validation data
        :return: the validation error
        """

        def is_in_interval(x, interval):
            return interval[0] <= x <= interval[1]

        errors = 0
        for x, y in validation_data:
            # Check if x is in any interval in the list, if so, predict 1, otherwise 0
            predicted_y = any(is_in_interval(x, interval) for interval in intervals_list)
            errors += (predicted_y != y)
        return errors / len(validation_data)

    def find_val_best_intervals(self, intervals_array, validation_data):
        """
        This function finds the best intervals from a given array of intervals
        :param intervals_array: the array of intervals
        :param validation_data: the validation data
        :return: the best intervals, the validation error, and the index of the best intervals
        """
        min_error = float('inf')  # Initialize the minimum error to infinity, so the first error will be smaller
        best_intervals = None
        best_index = -1

        for index, intervals_list in enumerate(intervals_array): # We need enumerate to get the index of the best interval
            error = self.calculate_validation_error(intervals_list, validation_data)
            if error < min_error:
                min_error = error # Update the minimum error
                best_intervals = intervals_list # Update the best intervals
                best_index = index # Update the index of the best intervals

        return best_intervals, min_error, best_index
    #################################


if __name__ == '__main__':
    ass = Assignment2()
    ass.experiment_m_range_erm(10, 100, 5, 3, 100)
    ass.experiment_k_range_erm(1500, 1, 10, 1)
    ass.experiment_k_range_srm(1500, 1, 10, 1)
    ass.cross_validation(1500)

