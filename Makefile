# CPLEX
LIBFORMAT=static_pic
CPLEXLIBDIR   = /home/gkatsirelos/pkgs/ilog/cplex/lib/x86-64_sles10_4.1/$(LIBFORMAT)
CONCERTLIBDIR = /home/gkatsirelos/pkgs/ilog/concert/lib/x86-64_sles10_4.1/$(LIBFORMAT)
CONCERTINCDIR = /home/gkatsirelos/pkgs/ilog/concert/include
CPLEXINCDIR   = /home/gkatsirelos/pkgs/ilog/cplex/include

#
#OPTFLAGS=-g
OPTFLAGS=-O3
CXXFLAGS=--std=c++11 -D __STDC_LIMIT_MACROS -D __STDC_FORMAT_MACROS -Wall -Wno-parentheses -Wextra \
	-I$(CPLEXINCDIR) -I$(CONCERTINCDIR) -DIL_STD \
	$(OPTFLAGS)
LDLIBS= -lboost_program_options \
	-L$(CPLEXLIBDIR) -lilocplex -lcplex -L$(CONCERTLIBDIR) -lconcert -lpthread
LDFLAGS=--static
CC=g++


all: uai-cplex

clean:
	rm -f uai-cplex *~ *.o
