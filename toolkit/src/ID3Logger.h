#ifndef _ID3LOGGER_H_
#define _ID3LOGGER_H_

#include <fstream>

class ID3Logger 
{
public:
	/**
	 * Gets the singleton instance of the logger
     * @return 
     */
	static ID3Logger& getInstance();

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

	/**
	 * Logs the number of nodes that were pruned
	 */
	void logPrunedNodes(size_t nodesPruned);

private:
    /**The output log file*/
    std::ofstream _logFile;

	/** The singleton instance */
	static ID3Logger _loggerInstance;

    /**
     * Constructor
     */
    ID3Logger();
};

#endif