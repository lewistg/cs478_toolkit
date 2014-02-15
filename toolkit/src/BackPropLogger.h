#ifndef _BACKPROPLOGGER_H_
#define _BACKPROPLOGGER_H_

#include <fstream>

/**
 * Stats for a single epoch of training
 */
struct EpochStats
{
	EpochStats(double tsAcc, double tsMse, double vsAcc, double vsMse, double testSetAcc = 0, double testSetMse = 0):
		_tsAcc(tsAcc),
		_tsMse(tsMse),
		_vsAcc(vsAcc),
		_vsMse(vsMse),
		_testSetAcc(testSetAcc),
        _testSetMse(testSetMse)
	{

	}

	double _tsAcc;
	double _tsMse;
	double _vsAcc;
	double _vsMse;
	double _testSetAcc;
    double _testSetMse;
};

class BackPropLogger
{
public:
    /**
     * Constructor
     */
    BackPropLogger();

    /**
     * Destructor
     */
    ~BackPropLogger();

	/**
	 * Logs the output
	 */
	void logStats(const EpochStats& stats);

	

private:
    /**The output log file*/
    std::ofstream _logFile;
};

#endif