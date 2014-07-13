# CPLEX
# CPLEXHOME=/home/gkatsirelos/pkgs/ilog
# CPLEXHOME=/Users/bhurley/Applications/IBM/ILOG/CPLEX_Studio1251
CPLEXHOME=/opt/ibm/ILOG/CPLEX_Studio126
LIBFORMAT=static_pic
CPLEXLIBDIR   = $(CPLEXHOME)/cplex/lib/x86-64_sles10_4.1/$(LIBFORMAT)
CONCERTLIBDIR = $(CPLEXHOME)/concert/lib/x86-64_sles10_4.1/$(LIBFORMAT)
CONCERTINCDIR = $(CPLEXHOME)/concert/include
CPLEXINCDIR   = $(CPLEXHOME)/cplex/include

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


all: uai-proteus

clean:
	rm -f uai-proteus *~ *.o
