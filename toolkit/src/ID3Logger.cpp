#include "ID3Logger.h"

ID3Logger::ID3Logger()
{
	_logFile.open("id3_log.txt", std::ios::trunc);
}

ID3Logger::~ID3Logger()
{
	_logFile.close();
}

void ID3Logger::logNodeEntropy(double entropy, size_t level)
{
	for(size_t i = 0; i < level; i++)
		_logFile << "\t";
	_logFile << "Node entropy: " << entropy << std::endl;
}

void ID3Logger::logSplitInfoGain(size_t attrSplitOn, double infoGain, size_t level)
{
	for(size_t i = 0; i < level; i++)
		_logFile << "\t";
	_logFile << "Split on " << attrSplitOn << " for info gain of " << infoGain << std::endl;
}