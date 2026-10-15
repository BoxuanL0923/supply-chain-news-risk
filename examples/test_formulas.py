"""Synthetic unit checks; no licensed data, third-party packages, or model calls."""
import math
import re
import unittest


def log_rv(returns, window, floor=1e-12):
    values = returns[-window:]
    if len(values) != window or not all(math.isfinite(x) for x in values):
        raise ValueError('Full finite window required')
    return math.log(max(sum(x*x for x in values)/window, floor))


def future_target(returns):
    if len(returns) != 5 or not all(math.isfinite(x) for x in returns):
        raise ValueError('Five finite future returns required')
    value = sum(x*x for x in returns)/5
    if value <= 0:
        raise ValueError('All-zero target is undefined')
    return math.log(value)


class FormulaTests(unittest.TestCase):
    def test_decimal_units(self):
        self.assertAlmostEqual(log_rv([.01]*22, 22), math.log(.0001))
        self.assertAlmostEqual(math.sqrt(math.exp(log_rv([.01]*5, 5))), .01)

    def test_trailing_window(self):
        self.assertAlmostEqual(log_rv([.30, .01, -.02, .03, -.04, .05], 5),
                               math.log(sum(x*x for x in [.01, -.02, .03, -.04, .05])/5))

    def test_floor_only_for_inputs(self):
        self.assertAlmostEqual(log_rv([0]*22, 22), math.log(1e-12))
        with self.assertRaises(ValueError):
            future_target([0]*5)

    def test_response(self):
        self.assertAlmostEqual(future_target([.01]*5), math.log(.0001))
        with self.assertRaises(ValueError):
            future_target([.01]*4)

    def test_score_units(self):
        score = math.log(2)
        self.assertAlmostEqual(math.exp(score), 2)
        self.assertAlmostEqual(math.exp(score/2), math.sqrt(2))

    def test_log_count(self):
        self.assertEqual(math.log1p(0), 0)
        self.assertAlmostEqual(math.log1p(3), math.log(4))

    def test_historical_cash_regex_limitation(self):
        pattern = re.compile(r'(?:acquir|buy|takeover|acquisition|merger).*\b(?:cash|agreement|agrees)\b|all-cash', re.I)
        self.assertIsNone(pattern.search('Example agrees to acquisition'))
        self.assertIsNotNone(pattern.search('Example acquisition agreement signed'))


if __name__ == '__main__':
    unittest.main(verbosity=2)
