// -*- Mode: c++; c-basic-offset: 4 -*-

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <tuple>
#include <map>
#include <unordered_map>
#include <boost/format.hpp>
#include <boost/program_options.hpp>

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
    int var;
    int val;
};

struct wcsp {
    Cost ub;

    vector<size_t> domains;
    size_t nvars() const { return domains.size(); }

    vector<wcspfunc> functions;

    vector<assignment> evidence;
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

    is >> nfun;
    for(size_t i = 0; i != nfun; ++i)
        w.functions.push_back(read_fun_scope(is));
    for(size_t i = 0; i != nfun; ++i)
        read_fun_matrix(w, w.functions[i], is);

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

bool new_incumbent = false;
bool should_print_incumbent = false;
bool alarm_expired = false;
bool first_solution = true;
vector<int> incumbent;

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
                IloNum val = cplex.getValue(vars.d[var][0]);
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
         fabs(lastIncumbent - getIncumbentObjValue())
         > 1e-5*(1.0 + fabs(getIncumbentObjValue())) ) {
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
                        incumbent.push_back(val);
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
    }

    if( getCplexTime() - lastPrint > 10.0 ) {
        should_print_incumbent = true;
        lastPrint = getCplexTime();
    }

    if( new_incumbent && should_print_incumbent ) {
        print_incumbent(ofs);
        lastPrint = getCplexTime();
    }
}

enum encoding { TUPLE, DIRECT, MIXED };

void solveilp(wcsp const& w, encoding enc, ostream& ofs, double timeout)
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

    cplex.extract(ilomodel);

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

    if( new_incumbent )
        print_incumbent(ofs);
    else if( first_solution ) {
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

    cout << "Solution status = " << cplex.getStatus() << endl;
    cout << "Solution value  = " << cplex.getObjValue()
         << ", log10like = " << -cplex.getObjValue()/log(10.0)
         << ", probability = " << exp(-cplex.getObjValue())
         << endl;
}

int main(int argc, char* argv[])
{
    namespace po = boost::program_options;

    string encoding;
    po::options_description desc("Allowed options");
    desc.add_options()
        ("help", "produce help message")
        ("encoding", po::value<string>(&encoding)->default_value("tuple"),
                "use direct/tuple encoding")
        ("input-file,i", po::value<string>(), "uai input file")
        ("evidence-file,e", po::value<string>(), "evidence file")
        ("query-file,q", po::value<string>(), "query file")
        ("task,t", po::value<string>(), "task")
        ;

    po::positional_options_description p;
    p.add("input-file", 1).add("evidence-file", 1)
        .add("query-file", 1).add("task", 1);

    po::variables_map vm;
    po::store(po::command_line_parser(argc, argv).
              options(desc).positional(p).run(), vm);
    po::notify(vm);

    if (vm.count("help")) {
        cout << desc << "\n";
        return 1;
    }

    enum encoding enc = TUPLE;
    if (encoding == "direct" )
        enc = DIRECT;
    else if (encoding == "mixed")
        enc = MIXED;

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

    wcsp w = readwcsp(ifs);
    read_evidence(w, efs);

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

    size_t dirpos = input_file.rfind('/');
    if(dirpos == string::npos )
        dirpos = 0;
    else
        ++dirpos;
    string result_file = input_file.substr(dirpos)+"."+task;
    ofstream ofs(result_file);
    if( !ofs ) {
        cout << "could not open " << result_file << "\n";
        return 1;
    }

    cout << cpuTime() << " to read input\n";
    const char *inftime = getenv("INF_TIME");
    if(!inftime) {
        cout << "INF_TIME not set\n";
        return 1;
    }
    double timeout = stoi(inftime) - cpuTime();

    cout << "Using " << encoding << " encoding\n";

    solveilp(w, enc, ofs, timeout);

    return 0;
}
