#ifndef _ID3LOGGER_H_
#define _ID3LOGGER_H_

#include <fstream>

class ID3Logger 
{
public:
    /**
     * Constructor
     */
    ID3Logger();

    /**
     * Destructor
     */
	~ID3Logger();

	/**
	 * Logs a node's entropy calculation
	 */
	void logNodeEntropy(double entropy, size_t level);

	/**
	 * Logs information gain of splitting on the given attribute
	 */
	void logSplitInfoGain(size_t attrSplitOn, double infoGain, size_t level);

private:
    /**The output log file*/
    std::ofstream _logFile;
};

#endif