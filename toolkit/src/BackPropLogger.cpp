#include "BackPropLogger.h"

BackPropLogger::BackPropLogger()
{
	_logFile.open("back_prop_log.txt", std::ios::trunc);
	_logFile << "test set accuracy, test set mean squared error,";
	_logFile << " validation set accuracy, valdiation set mean squared error,";
	_logFile << "test set accuracy";
	_logFile << std::endl;
}

BackPropLogger::~BackPropLogger()
{
	_logFile.close();
}

void BackPropLogger::logStats(const EpochStats& stats)
{
	_logFile << stats._tsAcc << ", ";
	_logFile << stats._tsMse << ", ";
	_logFile << stats._vsAcc << ", ";
	_logFile << stats._vsMse << ", ";
	_logFile << stats._testSetAcc << std::endl;
}