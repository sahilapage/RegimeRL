import unittest
from unittest.mock import MagicMock
from your_module import reward_function

class TestRewardFunction(unittest.TestCase):

    def test_reward_function_default(self):
        market_conditions = {
            'price': 100,
            'volume': 1000,
            'trend': 'up'
        }
        expected_reward = 10
        self.assertEqual(reward_function(market_conditions), expected_reward)

    def test_reward_function_high_price(self):
        market_conditions = {
            'price': 200,
            'volume': 1000,
            'trend': 'up'
        }
        expected_reward = 20
        self.assertEqual(reward_function(market_conditions), expected_reward)

    def test_reward_function_low_price(self):
        market_conditions = {
            'price': 50,
            'volume': 1000,
            'trend': 'up'
        }
        expected_reward = 5
        self.assertEqual(reward_function(market_conditions), expected_reward)

    def test_reward_function_high_volume(self):
        market_conditions = {
            'price': 100,
            'volume': 2000,
            'trend': 'up'
        }
        expected_reward = 15
        self.assertEqual(reward_function(market_conditions), expected_reward)

    def test_reward_function_low_volume(self):
        market_conditions = {
            'price': 100,
            'volume': 500,
            'trend': 'up'
        }
        expected_reward = 5
        self.assertEqual(reward_function(market_conditions), expected_reward)

    def test_reward_function_up_trend(self):
        market_conditions = {
            'price': 100,
            'volume': 1000,
            'trend': 'up'
        }
        expected_reward = 10
        self.assertEqual(reward_function(market_conditions), expected_reward)

    def test_reward_function_down_trend(self):
        market_conditions = {
            'price': 100,
            'volume': 1000,
            'trend': 'down'
        }
        expected_reward = -10
        self.assertEqual(reward_function(market_conditions), expected_reward)

    def test_reward_function_invalid_market_conditions(self):
        market_conditions = {
            'price': 'invalid',
            'volume': 1000,
            'trend': 'up'
        }
        with self.assertRaises(TypeError):
            reward_function(market_conditions)

    def test_reward_function_missing_market_conditions(self):
        market_conditions = {
            'price': 100,
            'volume': 1000
        }
        with self.assertRaises(KeyError):
            reward_function(market_conditions)

if __name__ == '__main__':
    unittest.main()