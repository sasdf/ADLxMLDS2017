import time
import copy
import json

logfile = open('output/logs/-%d.log' % time.time(), 'w')

def logging(metrics, others={}, save=True):
    if save:
        data = copy.copy(others)
        for m in metrics:
            data[m.name] = m.value()
        logfile.write(json.dumps(data) + '\n')
    return ', '.join(map(str, metrics))
