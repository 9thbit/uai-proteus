"""
    Takes objectives.csv and times.csv to compute scores.csv

    Computes a score which combines the relative objective qualtiy (robj) with
    the ratio of the time budget taken by the solver (rtime). The current
    scoring function is:

    exp(- (SCOREALPHA * robj + SCOREBETA * rtime))
"""


from proteustrain import read_csv
from math import exp
import csv
import os


csvfolder = "csv"
filenametime = os.path.join(csvfolder, "times.csv")
filenamescores = os.path.join(csvfolder, "scores.csv")
filenameobjs = os.path.join(csvfolder, "objectives.csv")
instancecol = "instance"  # Output
default_cutoff = 1200.0

# Scaling factors for score computation
SCOREALPHA = 2.0
SCOREBETA = 1.0


def computescore():
    solverlist, timedict = read_csv(filenametime)
    osolvernames, objsdict = read_csv(filenameobjs)
    fscore = open(filenamescores, "wt")

    headers = [instancecol] + solverlist
    writerscore = csv.DictWriter(fscore, headers)
    writerscore.writeheader()

    nrsolved, nrunsolved = 0, 0
    for instance, timerow in timedict.iteritems():
        objrow = objsdict[instance]
        dscore = {instancecol: instance}

        validvalues = [x for x in objrow if x < float('inf')]
        if validvalues:
            best, worst = min(validvalues), max(validvalues)
            nrsolved += 1
            objdenom = best - worst
            for i, s in enumerate(solverlist):
                obj = objrow[i]
                if obj < float('inf'):
                    robj = 0.0
                    if objdenom != 0.0:
                        robj = (best - obj) / objdenom
                    rtime = min(timerow[i], default_cutoff) / default_cutoff
                    dscore[s] = exp(- (SCOREALPHA * robj + SCOREBETA * rtime))
                else:
                    dscore[s] = -1.0
        else:  # Nobody solved the instance
            nrunsolved += 1
            for s in solverlist:
                dscore[s] = 0.0

        writerscore.writerow(dscore)

    print "%d instances were solved by at least one solver." % nrsolved
    print "%d instances were not solved by any solver." % nrunsolved


if __name__ == '__main__':
    computescore()
