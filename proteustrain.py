from sklearn import tree, cross_validation, ensemble
from sklearn.utils import check_random_state
from sklearn.base import BaseEstimator, ClassifierMixin
from collections import OrderedDict, Counter
from itertools import combinations
import numpy as np
import cStringIO
import csv
import sys


NUM_CV_FOLDS = 10
FEATPREDTHRESHOLD = 90.0  # Time budget for feature computation
predictioncodefilename = "proteuspredictors.h"
featname = None


def read_csv(filename, mapvalues=float, instancelist=None,
             excludedcolumns=None):
    instances = OrderedDict()
    with open(filename, "rt") as f:
        reader = csv.DictReader(f)
        fieldnames = [x for x in reader.fieldnames if x != "instance"]
        if excludedcolumns:
            for x in excludedcolumns:
                if x in fieldnames:
                    fieldnames.remove(x)
        mydict = {}
        for row in reader:
            d = [mapvalues(row[x]) for x in fieldnames]
            instance = str(row["instance"])
            if instancelist:
                mydict[instance] = d
            else:
                instances[instance] = d

    if instancelist:
        for instance in instancelist:
            instances[instance] = mydict[instance]

    return fieldnames, instances


def drawclftree(treefilename, c, featnames, tree_title):
    "Draws a fitted sklearn decision tree 'c' to a pdf 'treefilename'."
    import pydot

    dot_data = cStringIO.StringIO()
    tree.export_graphviz(c,
                         out_file=dot_data,
                         feature_names=featnames)
    dot_string = dot_data.getvalue()
    dot_string = dot_string[:dot_string.rindex("}")] + \
        """labelloc="t"; label="%s";}""" % tree_title
    graph = pydot.graph_from_dot_data(dot_string)
    graph.write_pdf(treefilename)


def extract_feattime_prediction_matrices(featnames,
                                         featuresdict,
                                         feattimedict,
                                         featpredthreshold=FEATPREDTHRESHOLD,
                                         ):
    """
        Extracts the feature matrix and label vector from the CSV data to numpy
        arrays for sklearn.
    """

    # The set of features that will be used to make the feattime prediction
    # predfeatures = ["filesize", "num_vars", "num_cfs"]
    predfeatures = ["filesize"]

    X, y = [], []
    for instance, row in feattimedict.iteritems():
        rowfeats = featuresdict[instance]
        label = int(row[0] > featpredthreshold or
                    rowfeats[featnames.index("num_vars")] == 0)

        X.append([rowfeats[featnames.index(name)] for name in predfeatures])
        y.append(label)

    return np.array(X), np.array(y)


def extract_solver_matrices(featuresdict, solvernames, scoresdict, timedict,
                            successdict):
    """
        Extracts the feature matrix and label vector from the CSV data to numpy
        arrays for sklearn.
    """
    X, y, scoresmatrix = [], [], []

    def minsolver(instancename):
        successrow = successdict[instancename]
        scorerow = scoresdict[instancename]

        best = -sys.maxint, None
        for i, solver in enumerate(solvernames):
            succ = successrow[i]
            s = scorerow[i]
            if succ >= 1.0 and s > best[0]:
                best = s, i
        return best

    mask = []
    for instance, featrow in featuresdict.iteritems():
        bestscore, bestsolver = minsolver(instance)
        mask.append(bool(bestsolver is not None))
        X.append(featrow)
        y.append(solvernames[bestsolver] if bestsolver is not None else "None")
        scoresmatrix.append(scoresdict[instance])

    return np.array(X), np.array(y), np.array(scoresmatrix), np.array(mask)


def extract_decision_tree(c, funcname):
    """
        Returns C++ code with a function named 'funcname' which implements the
        sklearn decision tree 'c'.
    """

    if not isinstance(c, tree.DecisionTreeClassifier):
        raise RuntimeError("not instance of decision tree %s" % str(type(c)))

    left = c.tree_.children_left
    right = c.tree_.children_right
    threshold = c.tree_.threshold
    features = c.tree_.feature
    numnodes = len(left)
    value = []
    for i, valuerow in enumerate(c.tree_.value):
        # labels are stored in reverse order at the leaf node by sklearn
        label0 = list(reversed(valuerow[0]))
        if left[i] == -1:
            nodevalue = np.argmax(label0)
            value.append(nodevalue)
        else:
            value.append(-1)

    # Sanity checks
    assert len(right) == numnodes
    assert len(threshold) == numnodes
    assert len(features) == numnodes
    assert len(value) == numnodes

    for i, v in enumerate(left):
        assert v >= -1
        assert v < numnodes
        if v == -1:
            assert value[i] >= 0
        else:
            assert value[i] == -1
    for i, v in enumerate(right):
        assert v >= -1
        assert v < numnodes

    funcstr = """
size_t %s(vector<double> features){
    int left[] = {%s};
    int right[] = {%s};
    int value[] = {%s};
    int featind[] = {%s};
    double threshold[] = {%s};

    size_t nid = 0;
    while(left[nid] != -1){
        if(features[featind[nid]] <= threshold[nid]) nid = left[nid];
        else nid = right[nid];
    }
    return (size_t)value[nid];
}
""" % (
        funcname,
        ",".join(map(str, left)),
        ",".join(map(str, right)),
        ",".join(map(str, value)),
        ",".join(map(str, features)),
        ",".join(map(str, threshold)),
    )
    return funcstr


def extract_randomforest(c, funcname):
    """
        Produces C++ code which implements the sklearn random forrest 'c'. The
        prediction funcion is named 'funcname' which returns a vector with the
        predicted labels ordered by the number of votes.
    """
    func = cStringIO.StringIO()
    N = len(c.classes_)
    estimators = []
    for i, t in enumerate(c.estimators_):
        name = "%s_%d" % (funcname, i)
        print >> func, extract_decision_tree(t, name)
        estimators.append(name)

    print >> func, "vector<size_t> %s(vector<double> features){" % funcname
    print >> func, "    int count[] = {%s};" % ",".join(["0"] * N)
    print >> func, """    vector<size_t> solverorder;
    for(size_t i=0; i<%d; i++) solverorder.push_back(i);""" % N

    for name in estimators:
        print >> func, "    count[%s(features)]++;" % name
    print >> func, """    cout << "%s Votes:";
    for(int i=0; i<%d; i++) cout << " " << count[i];
    cout << endl;""" % (funcname, N)

    print >> func, """
    // Sort the solverorder by vote count
    sort(solverorder.begin(), solverorder.end(),
       [&count](size_t i1, size_t i2) {return count[i1] > count[i2];});

    cout << "%s Order:";
    for(auto x : solverorder) cout << " " << x;
    cout << endl;
    return solverorder;
}
""" % (funcname)

    return func.getvalue()


def extract_randomforestsingle(c, funcname, classmap):
    """
        Produces C++ code which implements the sklearn random forrest 'c'. The
        prediction funcion is named 'funcname' which returns a map from the
        decision tree label to the number of votes by trees in the forrest.
        These are used by the pairwise classifier to tally votes between
        forrests.
    """
    func = cStringIO.StringIO()
    N = len(classmap)
    estimators = []
    for i, t in enumerate(c.estimators_):
        name = "%s_%d" % (funcname, i)
        print >> func, extract_decision_tree(t, name)
        estimators.append(name)

    print >> func, "map<int, int> %s(vector<double> features){" % funcname
    print >> func, "    int count[] = {%s};" % ",".join(["0"] * N)
    print >> func, "    size_t valuemap[] = {%s};" % \
        ",".join(map(str, classmap))

    for name in estimators:
        print >> func, "    count[%s(features)]++;" % name

    print >> func, """    cout << "%s %s Votes:";
    for(int i=0; i<%d; i++) cout << " " << count[i];
    cout << endl;""" % (funcname, "v".join(map(str, classmap)), N)

    print >> func, """
    map<int, int> m;
    for(int i=0; i<%d; i++) m[valuemap[i]] = count[i];
    return m;
}
""" % N

    return func.getvalue()


def extract_pairwiseclassifier(c, funcname):
    """
        Extracts C++ code implementing the custom PairwiseLabelClassifier. The
        resulting code, with function name 'funcname', returns a vector with
        predicted labels ordered by the number of votes from the random
        forrests.
    """
    func = cStringIO.StringIO()
    N = len(c.classes_)

    estimators = []
    for i, (c1, c2) in enumerate(combinations(c.classes_, 2)):
        e = c.estimators_[i]
        name = "%s_%d" % (funcname, i)
        classmap = [c.classes_.index(c1), c.classes_.index(c2)]
        print >> func, extract_randomforestsingle(e, name, classmap)
        estimators.append(name)

    print >> func, "vector<size_t> %s(vector<double> features){" % funcname
    print >> func, "    int count[] = {%s};" % ",".join(["0"] * N)
    print >> func, """    vector<size_t> solverorder;
    for(size_t i=0; i<%d; i++) solverorder.push_back(i);""" % N

    for i, name in enumerate(estimators):
        print >> func, "    map<int, int> m_%d = %s(features);" % (i, name)
        print >> func, \
            "    for(auto kv : m_%d) count[kv.first] += kv.second;" % i

    print >> func, """    cout << "%s Votes:";
    for(int i=0; i<%d; i++) cout << " " << count[i];
    cout << endl;""" % (funcname, N)

    print >> func, """
    // Sort the solverorder by vote count
    sort(solverorder.begin(), solverorder.end(),
       [&count](size_t i1, size_t i2) {return count[i1] > count[i2];});

    cout << "%s Order:";
    for(auto x : solverorder) cout << " " << x;
    cout << endl;
    return solverorder;
}
""" % (funcname)

    return func.getvalue()


def getpredictionscore(scores_test, predictions, solvernames):
    "Computes the score for given predictions. Used in CV."
    assert len(scores_test) == len(predictions)
    score = 0.0
    for scorerow, prediction in zip(scores_test, predictions):
        predictionind = solvernames.index(prediction)
        score += scorerow[predictionind]

    # Return the mean if we have different sized folds
    return score / len(scores_test)


def getproteusrandomforrest(X, y,
                            scoresmatrix,
                            solvernames,
                            featnames,
                            cvfolds=NUM_CV_FOLDS,
                            random_state=None,
                            ):
    """
        Trains a random forrest using stratified K-fold cross validation.
    """

    best = -sys.maxint, None
    cv = cross_validation.StratifiedKFold(y, n_folds=cvfolds)
    for i, (train_index, test_index) in enumerate(cv):
        X_train, X_test = X[train_index], X[test_index]
        y_train = y[train_index]
        scores_test = scoresmatrix[test_index]

        clf = RandomForrest99(random_state=random_state)
        # clf = OneR(random_state=random_state)  # A one rule decision tree
        clf.fit(X_train, y_train)
        predictions = clf.predict(X_test)

        score = getpredictionscore(scores_test, predictions, solvernames)
        if score > best[0]:
            print "New best", i, score
            best = score, clf

        # Uncomment this code to draw each tree
        # treefilename = "plots/decisiontree/dt_proteus_cv_%d.pdf" % i
        # tree_title = "Predict Solver - Fold %d" % i
        # drawclftree(treefilename, clf, featnames, tree_title)

    return best[1]


class OneR(tree.DecisionTreeClassifier):
    def __init__(self, *args, **kwargs):
        super(OneR, self).__init__(max_depth=1, *args, **kwargs)


class RandomForrest99(ensemble.RandomForestClassifier):
    def __init__(self, *args, **kwargs):
        super(RandomForrest99, self).__init__(n_estimators=99, *args, **kwargs)


class PairwiseLabelClassifier(BaseEstimator, ClassifierMixin):
    """
        Learns an ensemble of pairwise classifiers, each of which is an instance
        of 'estimator_class' which is given in the constructor. Expects the Y
        data to be a relative rankings of the labels.
    """

    def __init__(self,
                 classes,
                 estimator_class=RandomForrest99,
                 *args, **kwargs):
        """
            estimator_class is a class object which will be instantiated to
            train each of the pairwise models. additional arguments will be
            passed on to the estimator_class constructor.
        """
        self.classes_ = classes
        self.estimators_ = []
        self.estimator_class = estimator_class
        self.init_args = args
        self.init_kwargs = kwargs

    def fit(self, X, y):
        # For each pair of labels extract the training data for which the two
        # labels differ in ranking, not training on rows where they are
        # equivalent.
        for c1, c2 in combinations(self.classes_, 2):
            thisX, thisy = [], []
            for xrow, yrow in zip(X, y):
                v1 = yrow[self.classes_.index(c1)]
                v2 = yrow[self.classes_.index(c2)]
                if v1 > v2:
                    thisX.append(xrow)
                    thisy.append(c1)
                elif v2 > v1:
                    thisX.append(xrow)
                    thisy.append(c2)

            # print c1, c2, Counter(thisy)
            thisX, thisy = np.array(thisX), np.array(thisy)

            c = self.estimator_class(*self.init_args, **self.init_kwargs)
            c.fit(thisX, thisy)
            self.estimators_.append(c)

        return self

    def predict(self, X):
        # Count the number of votes by each estimaotr for each class
        votes = np.zeros((np.shape(X)[0], len(self.classes_)))
        for e in self.estimators_:
            e_predicts = e.predict(X)
            for i, pred_value in enumerate(e_predicts):
                votes[i][self.classes_.index(pred_value)] += 1

        # Majority, could be something else
        predicitonids = np.argmax(votes, 1)
        output = np.array([self.classes_[i] for i in predicitonids])
        return output


def produce_pairwiserandomforrest_predictor(
        featnames, featuresdict, feattimedict, solvernames, scoresdict,
        timedict, successdict, instancelist, featuresmask, filesizelimit,
        shorttimesolver,
        ):
    """
        Using K-Fold cross validation to build a pariwise random forrest
        classifier and output the C++ code.
    """

    X, scoresmatrix = [], []
    unsolvedmask = []
    for instance, featrow in featuresdict.iteritems():
        successrow = successdict[instance]
        scorerow = scoresdict[instance]
        X.append(featrow)
        scoresmatrix.append(scorerow)
        unsolvedmask.append(np.any(successrow))

    X = np.array(X)
    scoresmatrix = np.array(scoresmatrix)
    unsolvedmask = np.array(unsolvedmask)

    # Remove the instances that we couldn't compute features for or unsolved
    print "Lens before:", len(X), len(scoresmatrix), len(instancelist)
    excludemast = featuresmask & unsolvedmask
    X = X[excludemast]
    scoresmatrix = scoresmatrix[excludemast]
    instancelist = np.array(instancelist)[excludemast]
    print "Lens after:", len(X), len(scoresmatrix), len(instancelist)

    random_state = check_random_state(4242)  # Reproduce
    best = -sys.maxint, None
    cv = cross_validation.KFold(len(X),
                                n_folds=NUM_CV_FOLDS,
                                random_state=random_state)
    for i, (train_index, test_index) in enumerate(cv):
        X_train, X_test = X[train_index], X[test_index]
        scorematrix_train = scoresmatrix[train_index]
        scores_test = scoresmatrix[test_index]

        clf = PairwiseLabelClassifier(solvernames, random_state=random_state)
        clf.fit(X_train, scorematrix_train)
        predictions = clf.predict(X_test)

        score = getpredictionscore(scores_test, predictions, solvernames)
        if score > best[0]:
            print "New best", i, score
            best = score, clf

    writepredictioncode(best[1], filesizelimit, solvernames, shorttimesolver)


def produce_randomforrest_predictor(
        featnames, featuresdict, feattimedict, solvernames, scoresdict,
        timedict, successdict, instancelist, featuresmask, filesizelimit,
        shorttimesolver,
        ):
    """
        Using K-Fold cross validation to build a single random forrest
        classifier and output the C++ code.
    """

    # The label is just that of the best solver
    X, y, scoresmatrix, unsolvedmask = extract_solver_matrices(
        featuresdict,
        solvernames,
        scoresdict,
        timedict,
        successdict,
    )

    # Remove the instances that we couldn't compute features for or unsolved
    print "Lens before:", len(X), len(y), len(scoresmatrix), len(instancelist)
    excludemast = featuresmask & unsolvedmask
    X = X[excludemast]
    y = y[excludemast]
    scoresmatrix = scoresmatrix[excludemast]
    instancelist = np.array(instancelist)[excludemast]
    print "Lens after:", len(X), len(y), len(scoresmatrix), len(instancelist)
    print Counter(y)
    random_state = check_random_state(4242)  # Reproduce

    clf = getproteusrandomforrest(
        X, y, scoresmatrix, solvernames, featnames,
        random_state=random_state)

    writepredictioncode(clf, filesizelimit, solvernames, shorttimesolver)

    featimportfilename = "tblfeatimportances.tex"
    writefeatureimportances(clf, featnames, featimportfilename)


def texify(s):
    import re
    s = re.sub("(?P<c>_)", "\_", s)
    return s


def writefeatureimportances(clf, featnames, filename, num_important_feat=15):
    """
        Computes the mean feature importance of the trees in the random forest
        'clf' and outputs to a LaTeX table in 'filename'.
    """
    sumfeatimportance = sumfeatimportance = np.zeros([len(featnames)])
    for c in clf.estimators_:
        sumfeatimportance += c.feature_importances_
    sumfeatimportance /= len(clf.estimators_)

    with open(filename, "wt") as f:
        print >> f, "\\begin{tabular}[c]{lr} \\toprule"
        print >> f, "Feature & Gini Importance \\\\"
        print >> f, "\\midrule"

        feature_importances_dict = dict(zip(featnames, sumfeatimportance))
        for i, (k, v) in enumerate(sorted(feature_importances_dict.iteritems(),
                                   key=lambda (k, v): v, reverse=True)):
            if v <= 0.0 or i >= num_important_feat:
                print >> f, r"%",
            print >> f, "%s & %.5f \\\\" % (texify(k), v)
        print >> f, "\\bottomrule\n\\end{tabular}"


def writepredictioncode(clf, filesizelimit, solvernames, shorttimesolver):
    with open(predictioncodefilename, "wt") as f:
        print >> f, "using namespace std;"

        print >> f, "#define PROTEUSFILESIZELIMIT %d" % filesizelimit

        print >> f, "\n// Solver mapping from name to prediction ID"
        for i, solvername in enumerate(solvernames):
            print >> f, "#define proteus_%s %d" % (solvername, i)

        print >> f, "#define proteus_short_backup %d" % shorttimesolver

        predictname = "predictsolver"
        if isinstance(clf, tree.DecisionTreeClassifier):
            print >> f, extract_decision_tree(clf, predictname)
        elif isinstance(clf, ensemble.RandomForestClassifier):
            print >> f, extract_randomforest(clf, predictname)
        elif isinstance(clf, PairwiseLabelClassifier):
            print >> f, extract_pairwiseclassifier(clf, predictname)


def main():
    global featnames
    featfilename = "csv/features.csv"
    feattimesfilename = "csv/feattimes.csv"
    timesfilename = "csv/times.csv"
    scoresfilename = "csv/scores.csv"
    successfilename = "csv/success.csv"

    solvernames, scoresdict = read_csv(scoresfilename)
    tsolvernames, timedict = read_csv(timesfilename)
    ssolvernames, successdict = read_csv(successfilename)
    instancelist = scoresdict.keys()  # Keep a consitent order of the instances

    featnames, featuresdict = read_csv(featfilename, instancelist=instancelist)
    _, feattimedict = read_csv(feattimesfilename, instancelist=instancelist)

    # Assert that the keys match
    assert set(instancelist) == set(featuresdict.keys())
    assert set(instancelist) == set(feattimedict.keys())
    assert set(instancelist) == set(timedict.keys())
    assert set(instancelist) == set(successdict.keys())
    assert solvernames == tsolvernames
    assert solvernames == ssolvernames

    # Label instances for which feature computation timed-out or did not
    # succeed, these will be filtered later. Define filesize limit for this.
    X, y = extract_feattime_prediction_matrices(featnames,
                                                featuresdict,
                                                feattimedict)

    clf = OneR()
    clf.fit(X, y)
    filesizelimit = int(clf.tree_.threshold[0])
    # Mask to later use only instances for which we have features
    featuresmask = y == 0

    def compute_backupsolver(timelimit):
        shorttimescores = [0.0] * len(solvernames)
        for instance, timerow in timedict.iteritems():
            successrow = successdict[instance]
            scorerow = scoresdict[instance]
            for i in xrange(len(solvernames)):
                if successrow[i] and timerow[i] <= timelimit:
                    shorttimescores[i] += scorerow[i]
        return np.argmax(np.array(shorttimescores))

    # Find the short time solver, i.e. the one giving the highest score in 20s
    shorttimesolver = compute_backupsolver(20.0)

    # Produce a single random forrest model
    produce_randomforrest_predictor(
        featnames, featuresdict, feattimedict, solvernames, scoresdict,
        timedict, successdict, instancelist, featuresmask, filesizelimit,
        shorttimesolver,
    )

    # Produce a pairwise label model
    # produce_pairwiserandomforrest_predictor(
    #     featnames, featuresdict, feattimedict, solvernames, scoresdict,
    #     timedict, successdict, instancelist, featuresmask, filesizelimit,
    #     shorttimesolver,
    # )


if __name__ == '__main__':
    main()
