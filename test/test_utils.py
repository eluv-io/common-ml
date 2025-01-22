import argparse
import random
from quick_test_py import Tester
import os
from dataclasses import asdict

from common_ml.utils.dictionary import nested_update


def test_dictionary():
    def t1():
        d1 = {'a': 1, 'b': '2'}
        d2 = {'b': '2', 'c': 4}
        d3 = nested_update(d1, d2)
        assert d3 == {'a': 1, 'b': '2', 'c': 4}, d3
        return []
    return [t1]
    
def main():
    tester = Tester(os.path.dirname(__file__) + '/test_data')
    tester.register('test_dictionary', test_dictionary())
    if args.record:
        tester.record(args.tests)
    else:
        tester.validate(args.tests)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tests', nargs='+')
    parser.add_argument('--record', action='store_true')
    args = parser.parse_args()
    main()