import sys

from pyannote_handler import test

if __name__ == "__main__":
    if len(sys.argv) <= 1:
        test()
    elif '--auto' in sys.argv:
        if '--cgn' in sys.argv:
            test(custom=False, prefix='cgn_')
        else:
            test(custom=False)
    elif '--cgn' in sys.argv:
        test(prefix='cgn_')
