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

struct wcspfunc {
    vector<size_t> scope;
    vector< wcsptuple > specs;

    size_t arity() const { return scope.size(); }
};

struct wcsp {
    Cost ub;

    vector<size_t> domains;
    size_t nvars() const { return domains.size(); }

    vector<wcspfunc> functions;
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

    string name;
    size_t nvars;
    size_t nfun;

    string line;

    getline(is, line);
    if( line != "MARKOV" )
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

/***********************************************************************/

struct cplexvars {
    vector<IloIntVarArray> d; // d_i,r, indexed by i and then r
    vector<IloIntVarArray > p; // p_S,t, indexed by S (0..nfun) and
                               // then t (the position of the tuple in
                               // f.specs)
};

cplexvars construct_ip_common(wcsp& w, IloEnv iloenv, IloModel ilomodel)
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

void add_tuple_constraints(wcsp &w, cplexvars &v,
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
                supports[j][t.tup[j]] += v.p[i][q];
            }
        }

        // and now post the completed thing
        for(size_t q = 0; q != f.scope.size(); ++q) {
            auto var = f.scope[q];
            for(size_t val = 0; val != w.domains[var]; ++val)
                ilomodel.add(supports[q][val] == 0);
        }
    }
}

void add_direct_constraints(wcsp &w, cplexvars &v,
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

void build_objective(wcsp &w, cplexvars &v,
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
                expr += f.specs[q].cost * v.p[i][q];
            }
        }
    }
    IloObjective obj(iloenv, expr, IloObjective::Minimize);
    ilomodel.add(obj);
}

enum encoding { TUPLE, DIRECT, MIXED };

void solveilp(wcsp& w, encoding enc)
{
    IloEnv iloenv;
    IloModel ilomodel(iloenv);

    cplexvars v = construct_ip_common(w, iloenv, ilomodel);
    if( enc != DIRECT )
        add_tuple_constraints(w, v, iloenv, ilomodel);
    if( enc != TUPLE )
        add_direct_constraints(w, v, iloenv, ilomodel);
    build_objective(w, v, iloenv, ilomodel);

    IloCplex cplex(iloenv);
    cplex.extract(ilomodel);

    cplex.solve();

    cout << "Solution status = " << cplex.getStatus() << endl;
    cout << "Solution value  = " << cplex.getObjValue() << endl;
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
        ("input-file,i", po::value<string>(), "wcsp input file")
        ;

    po::positional_options_description p;
    p.add("input-file", 1).add("output-file", 1);

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

    ifstream ifs(vm["input-file"].as<string>());
    if( !ifs ) {
        cout << "could not open " << argv[1] << "\n";
        return 1;
    }

    wcsp w = readwcsp(ifs);

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

    solveilp(w, enc);

    return 0;
}
