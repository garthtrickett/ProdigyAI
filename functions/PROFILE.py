"""
Profile's the burst mode pipeline and prints information on computational cost and frequency of function calls.

@author: Jeffrey Wardman

Created on 2nd July, 2019.
"""

import os
import io
import pstats
import cProfile
import spartan.binance.binance_get_symbol_splits


def profile_to_csv(filename='profile'):
    """Profiles and then saves the performance results in a CSV file.

    From:
    https://stackoverflow.com/a/51541290/10345128 and
    https://stackoverflow.com/a/52304147/10345128
    """
    profile = cProfile.Profile()
    profile.enable()

    spartan.binance.binance_get_symbol_splits.main()
    profile.disable()

    results = io.StringIO()
    stats = pstats.Stats(profile, stream=results).sort_stats('tottime')
    stats.print_stats()

    results = results.getvalue()
    # chop the string into a csv-like buffer
    results = 'ncalls' + results.split('ncalls')[-1]
    results = '\n'.join([','.join(line.rstrip().split(None, 6)) for line in results.split('\n')])

    # Create save directory if it does not already exist.
    path = os.path.join(os.getcwd(), 'profiles')
    if not os.path.isdir(path):
        os.makedirs(path)

    # Save to disk.
    if type(filename.rstrip('.')) == str:
        f = open(os.path.join(path, filename + '.csv'), 'w')
    else:
        f = open(os.path.join(path, filename.rsplit('.')[0] + '.csv'), 'w')
    f.write(results)
    f.close()

    # From http://www.blog.pythonlibrary.org/2014/03/20/python-102-how-to-profile-your-code/
    print('Rows are: '
          '\nncalls: number of calls made,'
          '\ntottime: total time spent in given function,'
          '\npercall = tottime/ncalls,'
          '\ncumtime is the cumulative time spent in this and all subfunctions,'
          '\nSecond percall column = cumtime/primitive calls,'
          '\nfilename:lineno(function) provides the respective data of each function.'
          )


if __name__ == '__main__':
    print('To use: import required file and change "file" to the correct file and function between profile.enable and disable.')
    file = os.path.join(os.getcwd(), 'binance', 'binance_get_symbol_splits.py')
    filename = os.path.splitext(file)[0].split(os.sep)[-1]
    profile_to_csv(filename='{}_profile'.format(filename))
