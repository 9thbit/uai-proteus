// -*- Mode: c++; c-basic-offset: 4 -*-

#include <iostream>
#include <fstream>
#include <sstream>
#include <numeric>
#include <vector>
#include <string>
#include <tuple>
#include <map>
#include <unordered_map>
#include <boost/format.hpp>
#include <boost/program_options.hpp>
#include "proteuspredictors.h"

#include <ilconcert/ilomodel.h>
#include <ilconcert/iloalg.h>
#include <ilcplex/ilocplex.h>

#include <sys/time.h>
#include <sys/resource.h>
#include <unistd.h>

double cpuTime(void)
{
    struct rusage ru;
    getrusage(RUSAGE_SELF, &ru);
    return (double)ru.ru_utime.tv_sec + (double)ru.ru_utime.tv_usec / 1000000 +
        (double)ru.ru_stime.tv_sec + (double)ru.ru_stime.tv_usec / 1000000;
}

using namespace std;
using boost::format;

// If at least this much time is remaining, we will try to verify the solution
// of a solver is feasible. If it is not, we will try the next solver.
#define VERIFYSOLLIMIT 3.0

template<typename T>
ostream& operator<<(ostream& os, vector<T> const& v)
{
    os << "v{";
    copy(begin(v), end(v), ostream_iterator<T>(os, ", "));
    os << "}";
    return os;
}


// trim from end
static inline std::string &rtrim(std::string &s) {
    s.erase(std::find_if(s.rbegin(), s.rend(),
                         [](int x) { return !std::isspace(x); }).base(),
            s.end());
    return s;
}


/***********************************************************************/

const bool debug = true;

typedef double Cost;

/***********************************************************************/


struct up{};

struct wcsptuple {
    vector<size_t> tup;
    Cost cost;

    wcsptuple(vector<size_t> const &t, Cost c) : tup(t), cost(c) {}
};

ostream& operator<<(ostream& os, wcsptuple const& t)
{
    os << t.tup << ":" << t.cost;
    return os;
}

struct wcspfunc {
    vector<size_t> scope;
    vector< wcsptuple > specs;

    size_t arity() const { return scope.size(); }
};

struct assignment {
    unsigned var;
    unsigned val;
};

struct wcsp {
    Cost ub;

    vector<size_t> domains;
    vector<size_t> degree;
    size_t nvars() const { return domains.size(); }

    vector<wcspfunc> functions;

    vector<assignment> evidence;
    vector<assignment> start;
};

template<typename T>
vector<T> read_vec(istream& is, int num)
{
    vector<T> r;
    T s;
    is >> s;
    while(is && num) {
        r.push_back(s);
        if( --num)
            is >> s;
    }
    return r;
}

wcspfunc read_fun_scope(istream& is)
{
    size_t arity;
    is >> arity;
    vector<size_t> scope = read_vec<size_t>(is, arity);
    if( arity != scope.size() )
        throw up();
    return {scope, {}};
}

template<typename Func>
void for_each_cartesian_product(vector<size_t> &ranges, Func f)
{
    vector<size_t> tuple(ranges.size(), 0);
    int i = 0;
    tuple[0] = 0;
    for(;;) {
        if( i == ranges.size() ) {
            f(tuple);
            --i;
            ++tuple[i];
        } else {
            if( tuple[i] >= ranges[i] ) {
                --i;
                if( i < 0 )
                    return;
                ++tuple[i];
            } else {
                ++i;
                if( i < ranges.size() )
                    tuple[i] = 0;
            }
        }
    }
}

void read_fun_matrix(wcsp& w, wcspfunc& f, istream& is)
{
    vector<size_t> doms;
    for(size_t var : f.scope)
        doms.push_back(w.domains[var]);

    int nt = 0;
    is >> nt;
    int expected = accumulate(begin(doms), end(doms), 1,
                              [](size_t i, size_t a) { return a*i; });
    if( expected != nt )
        throw up();

    for_each_cartesian_product(doms, [&](vector<size_t>& tuple) {
            double prob;
            is >> prob;
            f.specs.emplace_back( tuple, -log(prob) );
        });
}

wcsp readwcsp(istream& is)
{
    wcsp w;
    w.ub = numeric_limits<Cost>::max();

    string name;
    size_t nvars;
    size_t nfun;

    string line;

    getline(is, line);
    if( rtrim(line) != "MARKOV" )
        throw up();

    is >> nvars;
    w.domains = read_vec<size_t>(is, nvars);
    if( w.domains.size() != nvars )
        throw up();
    w.degree.resize(nvars, 0);

    is >> nfun;
    for(size_t i = 0; i != nfun; ++i)
        w.functions.push_back(read_fun_scope(is));
    for(size_t i = 0; i != nfun; ++i)
        read_fun_matrix(w, w.functions[i], is);

    for( auto& f: w.functions ) {
        if( f.scope.size() >= 2 )
            for(auto var: f.scope)
                ++w.degree[var];
    }

    return w;
}

void read_evidence(wcsp& w, istream& is)
{
    int nast;
    is >> nast;
    for(int i = 0; i != nast; ++i) {
        assignment a;
        is >> a.var >> a.val;
        w.evidence.push_back(a);
    }
}

void read_start(wcsp& w, istream& is)
{
    int solution=0;
    while(true) {
        string s;
        is >> s;
        if( !is )
            break;
        ++solution;
        unsigned nast;
        is >> nast;
        if( nast != w.nvars() ) {
            cout << "start file nvars not correct in solution " << solution << "\n";
            throw up();
        }
        w.start.clear(); // forget the last solution
        for(unsigned i = 0; i != nast; ++i) {
            assignment a;
            a.var = i;
            is >> a.val;
            w.start.push_back(a);
        }
    }
    cout << "Read " << solution << " starts, keeping the last only\n";
}


/***********************************************************************/

struct cplexvars {
    vector<IloIntVarArray> d; // d_i,r, indexed by i and then r
    vector<IloIntVarArray > p; // p_S,t, indexed by S (0..nfun) and
                               // then t (the position of the tuple in
                               // f.specs)
};

cplexvars construct_ip_common(wcsp const& w, IloEnv iloenv, IloModel ilomodel)
{
    cplexvars v;
    v.d.resize(w.nvars());
    for(size_t i = 0; i != w.nvars(); ++i) {
        if( w.domains[i] == 1 ) continue;
        else if(w.domains[i] == 2) {
            v.d[i] = IloIntVarArray(iloenv);
            v.d[i].add(IloIntVar(iloenv, 0, 1));
            if( debug ) {
                ostringstream oss;
                oss << "db(" << i << ")";
                v.d[i][0].setName(oss.str().c_str());
            }
        } else {
            IloIntArray lbs(iloenv, w.domains[i]), ubs(iloenv, w.domains[i]);
            for(size_t j = 0; j != w.domains[i]; ++j) {
                lbs[j] = 0;
                ubs[j] = 1;
            }
            v.d[i] = IloIntVarArray(iloenv, lbs, ubs);
            if( debug ) {
                ostringstream oss;
                oss << "d_" << i << "_";
                v.d[i].setNames(oss.str().c_str());
            }

            // we reuse ubs here as coefficients, since they are all 1
            ilomodel.add( IloScalProd(ubs, v.d[i]) == 1 );
        }
    }

    v.p.resize(w.functions.size());
    for(size_t i = 0; i != w.functions.size(); ++i) {
        wcspfunc const& f = w.functions[i];
        if( f.scope.size() == 1 )
            continue; // no need to introduce variables, just put the
                      // thing in the objective
        IloIntArray lbs(iloenv, f.specs.size()), ubs(iloenv, f.specs.size());
        for(size_t j = 0; j != f.specs.size(); ++j) {
            lbs[j] = 0;
            ubs[j] = 1;
        }
        v.p[i] = IloIntVarArray(iloenv, lbs, ubs);
        if( debug ) {
            ostringstream oss;
            oss << "p(" << i << ")";
            v.p[i].setNames(oss.str().c_str());
        }

        // we reuse ubs here as coefficients, since they are all 1
        ilomodel.add( IloScalProd(ubs, v.p[i]) == 1 );
    }

    return v;
}

void add_tuple_constraints(wcsp const&w, cplexvars const&v,
                           IloEnv iloenv, IloModel ilomodel)
{
    for(size_t i = 0; i != w.functions.size(); ++i) {
        wcspfunc const& f = w.functions[i];
        if( f.scope.size() == 1 )
            continue;

        // create all the nd constraints
        vector< vector<IloExpr> > supports(f.scope.size());
        for(size_t q = 0; q != f.scope.size(); ++q) {
            auto var = f.scope[q];
            switch( w.domains[var] ) {
            case 1:  continue;
            case 2:
                supports[q].resize(w.domains[var]);
                supports[q][0] = IloExpr(iloenv);
                supports[q][0] -= (1 - v.d[var][0]);
                supports[q][1] = IloExpr(iloenv);
                supports[q][1] -= v.d[var][0];
                break;
            default:
                supports[q].resize(w.domains[var]);
                for(size_t val = 0; val != w.domains[var]; ++val) {
                    supports[q][val] = IloExpr(iloenv);
                    supports[q][val] -= v.d[var][val];
                }
                break;
            }
        }

        // add the supports to each constraint
        for(size_t q = 0; q != f.specs.size(); ++ q) {
            auto& t = f.specs[q];
            for( size_t j = 0; j != f.scope.size(); ++j) {
                if( w.domains[f.scope[j]] >= 2 )
                    supports[j][t.tup[j]] += v.p[i][q];
            }
        }

        // and now post the completed thing
        for(size_t q = 0; q != f.scope.size(); ++q) {
            auto var = f.scope[q];
            if( w.domains[var] == 1 )
                continue;
            for(size_t val = 0; val != w.domains[var]; ++val)
                ilomodel.add(supports[q][val] == 0);
        }
    }
}

void add_direct_constraints(wcsp const&w, cplexvars const&v,
                            IloEnv iloenv, IloModel ilomodel)
{
    for(size_t i = 0; i != w.functions.size(); ++i) {
        wcspfunc const& f = w.functions[i];
        if( f.scope.size() == 1 )
            continue;
        for(size_t q = 0; q != f.specs.size(); ++ q) {
            auto& t = f.specs[q];
            IloExpr e(iloenv);
            e += v.p[i][q];
            for( size_t j = 0; j != f.scope.size(); ++j) {
                auto var = f.scope[j];
                switch(w.domains[var]) {
                case 1: continue;
                case 2:
                    if( t.tup[j] == 1 )
                        e += (1 - v.d[var][0]);
                    else
                        e += v.d[var][0];
                    break;
                default:
                    e += (1 - v.d[var][t.tup[j]]);
                    break;
                }
            }
            ilomodel.add(e >= 1);
        }
    }
}

void build_objective(wcsp const &w, cplexvars const &v,
                     IloEnv iloenv, IloModel ilomodel)
{
    IloExpr expr(iloenv);
    for(size_t i = 0; i != w.functions.size(); ++i) {
        wcspfunc const& f = w.functions[i];
        if( f.scope.size() == 1 ) {
            auto var = f.scope[0];
            if( w.domains[var] == 1 ) {
                continue;
            } else if( w.domains[var] == 2 ) {
                auto c0 = f.specs[0].cost, c1 = f.specs[1].cost;
                if( std::isinf(c0) ) {
                    c0 = 0;
                    ilomodel.add( v.d[var][0] == 1 );
                }
                if( std::isinf(c1) ) {
                    c1 = 0;
                    ilomodel.add( v.d[var][0] == 0 );
                }
                expr += c0 * (1 - v.d[var][0]) +
                    c1 * v.d[var][0];
            } else {
                for(size_t q = 0; q != f.specs.size(); ++q) {
                    auto &tuple = f.specs[q].tup;
                    auto cost = f.specs[q].cost;
                    if( std::isinf(cost) ) {
                        cost = 0;
                        ilomodel.add( v.d[var][tuple[0]] == 0 );
                    }
                    expr += cost * v.d[var][tuple[0]];
                }
            }
        } else {
            // non-unary CF
            for(size_t q = 0; q != f.specs.size(); ++q) {
                auto cost = f.specs[q].cost;
                if( std::isinf(cost) ) {
                    cost = 0;
                    ilomodel.add( v.p[i][q] == 0 );
                }
                expr += cost * v.p[i][q];
            }
        }
    }
    IloObjective obj(iloenv, expr, IloObjective::Minimize);
    ilomodel.add(obj);
}

void add_evidence(wcsp const& w, cplexvars const& v,
                  IloModel model)
{
    for(auto&& ast:w.evidence) {
        switch( w.domains[ast.var] ) {
        case 1:
            break;
        case 2:
            model.add(v.d[ast.var][0] == ast.val);
            break;
        default:
            model.add( v.d[ast.var][ast.val] == 1 );
            break;
        }
    }
}

void add_start(wcsp const& w, cplexvars const& v,
               IloCplex cplex)
{
    cout << "Adding MIP start\n";
    IloEnv env = cplex.getEnv();
    IloNumVarArray startVar(env);
    IloNumArray startVal(env);
    vector<size_t> asgn(w.nvars());
    for(auto&& ast: w.start) {
        asgn[ast.var] = 0;
        switch(w.domains[ast.var]) {
        case 1:
            break;
        case 2:
            if( w.degree[ast.var] >= 1 ) {
                startVar.add(v.d[ast.var][0]);
                startVal.add(ast.val);
            }
            break;
        default:
            for(size_t i = 0; i != w.domains[ast.var]; ++i) {
                startVar.add(v.d[ast.var][i]);
                startVal.add( ast.val == i );
            }
            break;
        }
    }

    cplex.addMIPStart(startVar, startVal);
}

void add_start_as_constraints(wcsp const& w, cplexvars const& v,
                              IloModel model)
{
    for(auto&& ast: w.start) {
        switch(w.domains[ast.var]) {
        case 1:
            break;
        case 2:
            model.add(v.d[ast.var][0] == ast.val);
            break;
        default:
            for(size_t i = 0; i != w.domains[ast.var]; ++i)
                model.add(v.d[ast.var][i] == (i==ast.val) );
            break;
        }
    }
}

bool new_incumbent = false;
bool should_print_incumbent = false;
bool alarm_expired = false;
bool first_solution = true;
vector<int> incumbent;
Cost incumbent_value = std::numeric_limits<Cost>::infinity();

void print_incumbent(ostream& ofs)
{
    cout << "printing\n";
    if( first_solution )
        ofs << "MPE\n";
    else
        ofs << "-BEGIN-\n";
    ofs << incumbent.size();
    for(auto val : incumbent)
        ofs << ' ' << val;
    ofs << endl;

    should_print_incumbent = false;
    new_incumbent = false;
    first_solution = false;
}

Cost verify_incumbent(wcsp const& w)
{
    Cost obj = 0.0;
    for(auto&& f: w.functions) {
        for(auto&& t: f.specs) {
            bool it = true;
            for(unsigned i = 0; i != f.scope.size(); ++i)
                if( incumbent[f.scope[i]] != t.tup[i] ) {
                    it = false;
                    break;
                }
            if( it ) {
                obj += t.cost;
                break;
            }
        }
    }
    incumbent_value = obj;
    cout << "incumbent cost " << obj << "\n";
    return obj;
}

void extract_solution(IloCplex cplex, wcsp const& w, cplexvars const& vars)
{
    // copy-paste from the callback. Don't know how to abstract it
    cout << "new incumbent " << cplex.getObjValue() << "\n";
    incumbent.clear();
    for(size_t var = 0; var != w.nvars(); ++var) {
        switch(w.domains[var]) {
        case 1:
            incumbent.push_back(0);
            break;
        case 2:
            {
                IloNum val = cplex.getValue(vars.d[var][0]) > 0.5;
                incumbent.push_back(val);
                break;
            }
        default:
            {
                IloNumArray vals(vars.d[var].getEnv());
                cplex.getValues(vals, vars.d[var]);
                size_t sum = 0;
                for(size_t q = 0; q != w.domains[var]; ++q) {
                    sum += vals[q] > 0.5;
                    if( vals[q] > 0.5 ) {
                        incumbent.push_back(q);
                    }
                }
                if( sum != 1 )
                    throw up();
                vals.end();
                break;
            }
        }
    }
}

ILOMIPINFOCALLBACK5(loggingCallback,
                    wcsp const&, w,
                    cplexvars const&, vars,
                    IloNum, lastIncumbent,
                    IloNum, lastPrint,
                    ostream&, ofs)
{
    bool record_incumbent = false;

    if ( hasIncumbent() &&
         lastIncumbent > getIncumbentObjValue() ) {
        lastIncumbent = getIncumbentObjValue();
        record_incumbent = true;
        new_incumbent = true;
    }

    if ( record_incumbent ) {
        cout << "new incumbent " << getIncumbentObjValue() << "\n";
        incumbent.clear();
        for(size_t var = 0; var != w.nvars(); ++var) {
            switch(w.domains[var]) {
            case 1:
                incumbent.push_back(0);
                break;
            case 2:
                {
                    try {
                        IloNum val = getIncumbentValue(vars.d[var][0]);
                        incumbent.push_back(val > 0.5);
                    } catch( IloException ) {
                        // a disconnected binary variable where both
                        // values have the same cost will not be
                        // extracted (a disconnected non-binary
                        // variable will have the sum ... = 1
                        // constraint, so the 0/1 representation will
                        // be extracted)
                        incumbent.push_back(0);
                    }
                    break;
                }
            default:
                {
                    IloNumArray vals(vars.d[var].getEnv());
                    getIncumbentValues(vals, vars.d[var]);
                    size_t sum = 0;
                    for(size_t q = 0; q != w.domains[var]; ++q) {
                        sum += vals[q] > 0.5;
                        if( vals[q] > 0.5 ) {
                            incumbent.push_back(q);
                        }
                    }
                    if( sum != 1 )
                        throw up();
                    vals.end();
                    break;
                }
            }
        }

        /* Verify that the correct tuple variables are set to 1 */
        for(size_t i = 0; i != w.functions.size(); ++i) {
            auto&& f = w.functions[i];
            if( f.scope.size() < 2 )
                continue;
            for(size_t j = 0; j != f.specs.size(); ++j) {
                auto&& t = f.specs[j];
                bool it = true;
                for(unsigned i = 0; i != f.scope.size(); ++i)
                    if( incumbent[f.scope[i]] != t.tup[i] ) {
                        it = false;
                        break;
                    }
                int tval = getIncumbentValue( vars.p[i][j] ) > 0.5;
                if( tval != it )
                    throw up();
            }
        }
    }

    if( getCplexTime() - lastPrint > 10.0 ) {
        should_print_incumbent = true;
        lastPrint = getCplexTime();
    }

    if( new_incumbent && should_print_incumbent ) {
        print_incumbent(ofs);
        verify_incumbent(w);
        lastPrint = getCplexTime();
    }
}

enum encoding { TUPLE, DIRECT, MIXED };
enum task { SOLVE, VERIFY };

void solveilp(wcsp const& w, encoding enc, ostream& ofs, double timeout,
              task t = SOLVE)
{
    IloEnv iloenv;
    IloModel ilomodel(iloenv);
    IloCplex cplex(iloenv);

    double start = cplex.getCplexTime();

    cplexvars v = construct_ip_common(w, iloenv, ilomodel);
    if( enc != DIRECT )
        add_tuple_constraints(w, v, iloenv, ilomodel);
    if( enc != TUPLE )
        add_direct_constraints(w, v, iloenv, ilomodel);
    add_evidence(w, v, ilomodel);
    build_objective(w, v, iloenv, ilomodel);


    if( t == VERIFY ) {
        if( w.start.empty() )
            throw up();
        add_start_as_constraints(w, v, ilomodel);
    }

    cplex.extract(ilomodel);
    if( !w.start.empty() ) {
        if( w.start.size() != w.nvars() ) {
            cout << "start has the wrong number of variables\n";
            throw up();
        }
        add_start(w, v, cplex);
    }

    if( !debug ) {
        cplex.setParam(IloCplex::MIPDisplay, 0);
    }

    double used = cplex.getCplexTime() - start;
    cout << "Consumed " << used << " to construct the model\n";

    cplex.setParam(IloCplex::Threads, 1);
    cplex.setParam(IloCplex::TiLim, timeout-used-1);

    cplex.use(loggingCallback(iloenv, w, v, IloInfinity,
                              cplex.getCplexTime(),
                              ofs));

    try {
        cplex.solve();
    } catch(IloException e) {
        cout << "oops, cplex says " << e << "\n";
    }

    if( t == VERIFY && cplex.getStatus() == IloAlgorithm::Infeasible ) {
        cout << "infeasible, should check why\n";
        throw up();
    }


    if( new_incumbent )
        print_incumbent(ofs);
    if( first_solution || cplex.getObjValue() < incumbent_value ) {
        // print_incumbent has not been called because the callback is
        // (I think) not called for solutions found before presolve.
        // So it is possible we are feasible or even optimal but
        // new_incumbent is false
        switch( cplex.getStatus() ) {
        case IloAlgorithm::Optimal:
        case IloAlgorithm::Feasible:
            cout << "printing\n";
            extract_solution(cplex, w, v);
            print_incumbent(ofs);
            break;
        default:
            break;
        }
    }

    verify_incumbent(w);

    cout << "Solution status = " << cplex.getStatus() << endl;
    cout << "Solution value  = " << cplex.getObjValue()
         << ", log10like = " << -cplex.getObjValue()/log(10.0)
         << ", probability = " << exp(-cplex.getObjValue())
         << endl;
}

/***********************************************************************/

struct vecstats {
    double mean, stddev, coefvar, min, max;
};

ostream& operator<<(ostream &s, const vecstats &v){
    s << v.mean << " " << v.stddev << " " << v.coefvar << " " << v.min << " " << v.max;
    return s;
}

struct featstruct {
    vector<string> featnames;
    vector<double> features;
};

template<typename T>
vecstats compute_vector_stats(vector<T> v, function<double (T)> f){
    // Computes some statistics about the list of values given by f(x) for each
    // x in the vector v. e.g. f can be used to get the arity of a wcspfunc.
    // f should take a single parameter, the same data type as the elements in
    // the vector v and should return a double.

    // Call f(x) on each element of v so we can cache the values, much faster
    // than calling f(x) repeatedly.
    vector<double> xs;
    xs.reserve(v.size());
    for(auto x : v) xs.push_back(f(x));

    double sum = accumulate(xs.begin(), xs.end(), 0.0);
    double mean = (double)(sum) / (double)xs.size();
    double sumsqrddev = accumulate(xs.begin(), xs.end(), 0.0, [&](double tot, double e) { 
        return tot + ((e - mean) * (e - mean));
    });
    double stddev = sqrt(sumsqrddev / xs.size());
    double coefvar = mean != 0.0 ? stddev / mean : 0.0;

    double min = *min_element(xs.begin(), xs.end());
    double max = *max_element(xs.begin(), xs.end());

    return {mean, stddev, coefvar, min, max};
}

void write_csv_features(ostream &ofs, string probname, featstruct f){
    ofs << "instance";  // Header line
    for(auto s : f.featnames) ofs << "," << s; ofs << endl;

    ofs << probname;
    for(auto v : f.features) ofs << "," << setprecision(10) << v; ofs << endl;
}

featstruct compute_features(wcsp w, long filesize, double timeread, double timeub){
    featstruct feats;

    // Adds a single feature
    auto addfeat = [&](string name, double v) {
        feats.featnames.push_back(name);
        feats.features.push_back(v);
    };

    // Adds each item from the vecstats to the feature list with prefix
    auto addfeatstats = [&](string prefix, vecstats v) {
        addfeat(prefix + "_mean", v.mean);
        addfeat(prefix + "_stddev", v.stddev);
        addfeat(prefix + "_coefvar", v.coefvar);
        addfeat(prefix + "_min", v.min);
        addfeat(prefix + "_max", v.max);
    };

    size_t N = w.nvars();
    addfeat("filesize", (double)filesize);
    addfeat("timeread", timeread);
    addfeat("timeub", timeub);
    addfeat("num_vars", N);
    addfeat("num_cfs", w.functions.size());
    addfeat("ub_init", w.ub);

    vecstats dstats = compute_vector_stats<size_t>(w.domains, [](size_t d) {return (double)d;});
    addfeatstats("domsize", dstats);

    vecstats aritystats = compute_vector_stats<wcspfunc>(w.functions,
        [&](wcspfunc f) { return (double)(f.arity()); });
    addfeatstats("arity", aritystats);

    // Count of arities at indices: 0:unused, 1:unary, 2:binary, 3:ternary, 4:greater than ternary
    size_t aritycount[] = {0, 0, 0, 0, 0};
    for(auto f : w.functions){
        size_t a = f.arity();
        if(a >= 1 && a <= 3) aritycount[a]++;
        else if(a >= 4) aritycount[4]++;
    }

    // Density of unary, binary, and ternary cost functions, i.e. the fraction
    // of the total number of possible cost functions of each arity.
    double density_unary = (double) aritycount[1] / (double)N;
    double density_binary = (double) aritycount[2] / (double)(N * (N - 1));
    double density_ternary = (double) aritycount[3] / (double)(N * (N - 1) * (N - 2));
    addfeat("density1", density_unary);
    addfeat("density2", density_binary);
    addfeat("density3", density_ternary);

    // Ratio of CFs that have arity 4 or greater.
    // double arity4plus = (double)(w.functions.size() - count1 - count2 - count3) / (double) w.functions.size();
    double arity4plus = (double)aritycount[4] / (double)w.functions.size();
    addfeat("arity4plus", arity4plus);

    return feats;
}

bool checksolutionfileexists(wcsp w, string result_file){
    // Checks that the solution file exists and contains a non-infinite cost
    Cost c = numeric_limits<Cost>::max();
    ifstream msfs(result_file);
    if( !msfs ) {
        cout << "could not read solution file " << result_file << "\n";
        return false;
    }
    read_start(w, msfs);

    // FIXME: copied from main below
    if( !w.start.empty() ) {
        incumbent.resize(w.nvars());
        for(auto&& ast: w.start)
            incumbent[ast.var] = ast.val;

        c = verify_incumbent(w);
    }

    return !std::isinf(c);
}


int launch_solver(wcsp w, int solverid, string input_file, string evidence_file, string result_file, double timeout){
    string cmd;
    int retval = 0;
    double startelapsed = cpuTime();

    // -------------------- internal CPLEX interface --------------------
    if(solverid == proteus_cplexdirect || solverid == proteus_cplextuple){
        ofstream ofs(result_file);
        if( !ofs ) {
            cout << "could not open " << result_file << "\n";
            return 1;
        }

        if(solverid == proteus_cplexdirect){
            cout << "cplex direct" << endl;
            try {
                solveilp(w, DIRECT, ofs, timeout);
                // system(("rm " + result_file).c_str());  // FIXME debug
            } catch(IloException e) {
                retval = -2;
            } catch(up) {
                retval = -3;
            }

        } else if(solverid == proteus_cplextuple){
            cout << "cplex tuple" << endl;
            try {
                solveilp(w, TUPLE, ofs, timeout);
            } catch(IloException e) {
                retval = -2;
            } catch(up) {
                retval = -3;
            }
        }

    } else {
        // -------------------- External solver --------------------
        if(solverid == proteus_mplp2){
            // argv2[0] = "./mplp2";
            // retval = execv("./mplp2", (char **)argv2);
            cmd = "./mplp2 " + input_file + " " + evidence_file + " ignorequery MPE";
        } else
        if(solverid == proteus_toulbar2){
            // argv2[0] = "./toulbar2";
            // retval = execv("./toulbar2", (char **)argv2);
            cmd = "./toulbar2 " + input_file + " " + evidence_file + " ignorequery MPE";

        } else if(solverid == proteus_tb2incop){
            // argv2[0] = "./tb2incop";
            // retval = execv("./tb2incop", (char **)argv2);
            cmd = "./tb2incop " + input_file + " " + evidence_file + " ignorequery MPE";

        } else {
            cerr << "Unknown solverid to launch" << endl;
            exit(1);  // FIXME default to one solver for submission
        }

        cout << "Launching: " << cmd << endl;
        retval = system(cmd.c_str());

        // if(solverid == proteus_toulbar2) system(("rm " + result_file).c_str());  // FIXME debug
    }

    if(retval != 0) {
        cout << "Error launching solver " << solverid << " return value: " << retval << endl; // << ". retrying with fallback solver." << endl;
    } else {
        // If we have time remaining, check that the solver did give a solution
        double remainingtime = timeout - (cpuTime() - startelapsed);
        cout << "Remaining time: " << remainingtime << endl;
        if(remainingtime > VERIFYSOLLIMIT){  // FIXME
            bool validsol = checksolutionfileexists(w, result_file);
            if(!validsol){
                cout << "Solver finished but the solution could not be verified." << endl;
                retval = -5;
            }
        }

    }
    return retval;
}






/***********************************************************************/


int main(int argc, char* argv[])
{
    namespace po = boost::program_options;

    // string encoding;
    bool verify_only = false, feat_only = false;
    int retval;

    po::options_description desc("Allowed options");
    desc.add_options()
        ("help", "produce help message")
        // ("encoding", po::value<string>(&encoding)->default_value("tuple"), // Encoding decision will be made by ML
        //         "use direct/tuple encoding")
        ("verify", po::value<bool>(&verify_only)->default_value(false),
                "verify MIP start and exit")
        ("input-file,i", po::value<string>(), "uai input file")
        ("evidence-file,e", po::value<string>(), "evidence file")
        ("query-file,q", po::value<string>(), "query file")
        ("task,t", po::value<string>(), "task")
        ("mip-start-file,s", po::value<string>(), "MIP start")
        ("feat-file,f", po::value<string>(), "save instance features to this file")
        ("feat-only", po::value<bool>(&feat_only)->default_value(false), "compute features and exit")
        ;

    po::positional_options_description p;
    p.add("input-file", 1).add("evidence-file", 1)
        .add("query-file", 1).add("task", 1)
        .add("mip-start-file", 1);

    po::variables_map vm;
    po::store(po::command_line_parser(argc, argv).
              options(desc).positional(p).run(), vm);
    po::notify(vm);

    if (vm.count("help")) {
        cout << desc << "\n";
        return 1;
    }

    if(!vm.count("input-file")) {
        cout << "must specify input file\n";
        return 1;
    }
    if(!vm.count("evidence-file")) {
        cout << "must specify evidence file\n";
        return 1;
    }
    if(!vm.count("query-file")) {
        cout << "must specify query file to ignore\n";
        return 1;
    }
    if(!vm.count("task")) {
        cout << "must specify task\n";
        return 1;
    }

    string task = vm["task"].as<string>();
    if( task != "MPE" ) {
        cout << "nope, cannot do " << task << "\n";
        return 1;
    }

    string input_file = vm["input-file"].as<string>(),
        evidence_file = vm["evidence-file"].as<string>();
    ifstream ifs(input_file);
    if( !ifs ) {
        cout << "could not open " << input_file << "\n";
        return 1;
    }

    ifstream efs(evidence_file);
    if( !efs ) {
        cout << "could not open " << evidence_file << "\n";
    }

    size_t dirpos = input_file.rfind('/');
    if(dirpos == string::npos )
        dirpos = 0;
    else
        ++dirpos;
    string result_file = input_file.substr(dirpos)+"."+task;

    // Get the filesize for feature time-limit prediction
    auto begin = ifs.tellg();
    ifs.seekg(0, ios_base::end);
    long long filesize = (long long)(ifs.tellg() - begin);
    ifs.seekg(begin);

    // Check if we're being run for 20 seconds
    const char *inftimestr = getenv("INF_TIME");
    if(!inftimestr) {
        cout << "INF_TIME not set\n";
        return 1;
    }
    double inftime = (double) stoi(inftimestr), timeout;
    cout << "INF_TIME: " << inftime << endl;


    if(inftime <= 21.0 || filesize > PROTEUSFILESIZELIMIT){
        // Don't read the problem, just run the fallback solver
        if(filesize > PROTEUSFILESIZELIMIT)
            cout << "Filesize: " << filesize << endl;

        cout << "Fallback solver" << endl;
        timeout = inftime - cpuTime();
        retval = launch_solver(wcsp(), proteus_toulbar2, input_file, evidence_file, result_file, timeout);
        return retval;
    }


    wcsp w = readwcsp(ifs);
    read_evidence(w, efs);
    double timeread = cpuTime();

    Cost newtop = 0;
    for(auto const& f : w.functions) {
        if( f.specs.empty() ) { // empty function
            throw up();
        }
        auto me = std::max_element(f.specs.begin(), f.specs.end(),
                                   [&](wcsptuple const& m, wcsptuple const& c){
                                       return c.cost < w.ub &&
                                       (c.cost > m.cost ||
                                        m.cost >= w.ub);
                                   });
        newtop += me->cost;
    }
    if( newtop < w.ub )
        w.ub = newtop;
    double timeub = cpuTime() - timeread;

    if( vm.count("mip-start-file") ) {
        string ms = vm["mip-start-file"].as<string>();
        cout << "Reading MIP start " << ms << "\n";
        ifstream msfs(ms);
        if( !msfs ) {
            cout << "could not read " << ms << "\n";
            return 1;
        }
        read_start(w, msfs);
    }

    if( !w.start.empty() ) {
        incumbent.resize(w.nvars());
        for(auto&& ast: w.start)
            incumbent[ast.var] = ast.val;
        ofstream ofs(result_file);
        if( !ofs ) {
            cout << "could not open " << result_file << "\n";
            return 1;
        }
        print_incumbent(ofs);
    }

    if( verify_only ) {
        verify_incumbent(w);
        return 0;
    }

    cout << cpuTime() << " to read input\n";

    // -------------------- Feature computation --------------------
    double beforefeat = cpuTime();
    featstruct feats = compute_features(w, filesize, timeread, timeub);
    double tcomputefeat = cpuTime() - beforefeat;
    cout << "time to compute features: " << tcomputefeat << endl;

    if(vm.count("feat-file")){
        string featfilename = vm["feat-file"].as<string>();
        ofstream ffs(featfilename);
        if(!ffs){
            cerr << "Could not open feature file " << featfilename << endl;
            return 1;
        }
        write_csv_features(ffs, vm["input-file"].as<string>(), feats);
        if(feat_only) return 0;
    }


    // -------------------- Solver Prediction --------------------
    vector<size_t> solverorder = predictsolver(feats.features);

    for(auto solverid : solverorder){
        cout << "Will run solver " <<  solverid << endl;
        timeout = inftime - cpuTime();
        retval = launch_solver(w, solverid, input_file, evidence_file, result_file, timeout);
        cout << retval;
        if(retval == 0) break;
    }

    return 0;
}

