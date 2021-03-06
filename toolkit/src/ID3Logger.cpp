#include "ID3Logger.h"

ID3Logger ID3Logger::_loggerInstance;

ID3Logger& ID3Logger::getInstance()
{
	return _loggerInstance;
}

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

void ID3Logger::logSplitInfoGain(size_t attrSplitOn, const std::string& attrName, double infoGain, size_t level)
{
	for(size_t i = 0; i < level; i++)
		_logFile << "\t";
	_logFile << "Split on " << attrSplitOn << "-" << attrName << " for info gain of " << infoGain << std::endl;
}

void ID3Logger::logSplitOnLaplacian(size_t attrSplitOn, double laplacian, size_t level)
{
	for(size_t i = 0; i < level; i++)
		_logFile << "\t";
	_logFile << "Split on " << attrSplitOn << " for Laplacian of " << laplacian << std::endl;
}

void ID3Logger::logNumberOfInstances(size_t nInstances, size_t level)
{
	for(size_t i = 0; i < level; i++)
		_logFile << "\t";
	_logFile << "Number of instances to split: " << nInstances << std::endl;
}

void ID3Logger::logPrunedNodes(size_t nodesPruned)
{
	_logFile << "Number of nodes pruned: " << nodesPruned << std::endl;
}