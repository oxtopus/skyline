import time
import unittest2 as unittest
from mock import patch

import algorithms

class TestAlgorithms(unittest.TestCase):
    """
    Test all algorithms with a common, simple/known anomalous data set
    """
    
    def data(self, ts):
        """ 
        Mostly ones (1), with a final value of 1000 
        """
        timeseries = map(list, zip(map(float, range(int(ts)-86400, int(ts)+1)), [1]*86401))
        timeseries[-1][1]=1000
        timeseries[-2][1]=1
        timeseries[-3][1]=1
        return ts, timeseries
    
    def test_tail_avg(self):
        _, timeseries = self.data(time.time())
        self.assertEqual(algorithms.tail_avg(timeseries), 334)

    def test_grubbs(self):
        _, timeseries = self.data(time.time())
        self.assertTrue(algorithms.grubbs(timeseries))

    @patch.object(algorithms, 'time')
    def test_first_hour_average(self, timeMock):
        timeMock.return_value, timeseries = self.data(time.time())
        self.assertTrue(algorithms.first_hour_average(timeseries))

    def test_simple_stddev_from_moving_average(self):
        _, timeseries = self.data(time.time())
        self.assertTrue(algorithms.simple_stddev_from_moving_average(timeseries))

    def test_stddev_from_moving_average(self):
        _, timeseries = self.data(time.time())
        self.assertTrue(algorithms.stddev_from_moving_average(timeseries))

    def test_mean_subtraction_cumulation(self):
        _, timeseries = self.data(time.time())
        self.assertTrue(algorithms.mean_subtraction_cumulation(timeseries))
    
    @patch.object(algorithms, 'time')
    def test_least_squares(self, timeMock):
        timeMock.return_value, timeseries = self.data(time.time())
        self.assertTrue(algorithms.least_squares(timeseries))
    
    def test_histogram_bins(self):
        _, timeseries = self.data(time.time())
        self.assertTrue(algorithms.histogram_bins(timeseries))
    
    @patch.object(algorithms, 'time')
    def test_run_selected_algorithm(self, timeMock):
        timeMock.return_value, timeseries = self.data(time.time())
        self.assertTrue(algorithms.run_selected_algorithm(timeseries))

if __name__ == '__main__':
    unittest.main()
    