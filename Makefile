# CPLEX

ifeq ($(shell whoami),gkatsirelos)
	CPLEXHOME=/home/gkatsirelos/pkgs/ilog
	CPLEXARCH=x86-64_sles10_4.1
	x86-64_linux
else ifeq ($(shell whoami),bhurley)
	CPLEXHOME=/Users/bhurley/Applications/IBM/ILOG/CPLEX_Studio126
	CPLEXARCH=x86-64_osx
else ifeq ($(shell whoami),vagrant)
	CPLEXHOME=/opt/ibm/ILOG/CPLEX_Studio126
	CPLEXARCH=x86-64_linux
endif

LIBFORMAT=static_pic
CPLEXLIBDIR   = $(CPLEXHOME)/cplex/lib/$(CPLEXARCH)/$(LIBFORMAT)
CONCERTLIBDIR = $(CPLEXHOME)/concert/lib/$(CPLEXARCH)/$(LIBFORMAT)
CONCERTINCDIR = $(CPLEXHOME)/concert/include
CPLEXINCDIR   = $(CPLEXHOME)/cplex/include

#
#OPTFLAGS=-g
OPTFLAGS=-O3
CXXFLAGS=--std=c++11 -D __STDC_LIMIT_MACROS -D __STDC_FORMAT_MACROS -Wall -Wno-parentheses -Wextra \
	-I$(CPLEXINCDIR) -I$(CONCERTINCDIR) -DIL_STD -Wno-sign-compare \
	$(OPTFLAGS)
LDLIBS= -lboost_program_options -lboost_filesystem -lboost_system \
	-L$(CPLEXLIBDIR) -lilocplex -lcplex -L$(CONCERTLIBDIR) -lconcert -lpthread
LDFLAGS=--static


CC=g++


all: uai-proteus

clean:
	rm -f uai-proteus *~ *.o
