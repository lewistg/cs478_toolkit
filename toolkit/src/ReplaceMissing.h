#ifndef _REPLACEMISSING_H_
#define _REPLACEMISSING_H_

#include <vector>
#include "matrix.h"

struct ReplacementStrat
{
	/**
	 * Replaces the missing data in a vector
	 */
	virtual void operator()(std::vector<size_t>& attrToMissingReplacement, std::vector<double>& features)
	{
		for(size_t i = 0; i < features.size(); i++)
		{
			if(features[i] == UNKNOWN_VALUE)
				features[i] = attrToMissingReplacement[i];
		}
	}
};

/**
 * Replaces the missing values with a new attribute
 * @param features
 */
struct ReplaceWithAttribute: public ReplacementStrat
{
	/**
	 * Returns a matrix with how the data was replaced
     * @param features
     * @return 
     */
	std::vector<size_t> operator()(Matrix& features)
    {
		std::vector<size_t> attrsToUnknownEnums;
		std::string unknownName = "unknown";
        for(size_t attr = 0; attr < features.cols(); attr++)
		{
			size_t unknownEnum = features.addEnumValue(attr, unknownName);
			attrsToUnknownEnums.push_back(unknownEnum);
		}

		for(size_t i = 0; i < features.rows(); i++)
		{
			for(size_t j = 0; j < features.cols(); j++)
			{
				if(features[i][j] == UNKNOWN_VALUE)
					features[i][j] = attrsToUnknownEnums[j];
			}
		}

		return attrsToUnknownEnums;
    }

	virtual void operator()(std::vector<size_t>& attrToMissingReplacement, std::vector<double>& features)
	{
		ReplacementStrat::operator ()(attrToMissingReplacement, features);
	}
};

/**
 * Replaces the missing data with the most common value
 * @param features
 */
struct ReplaceWithMode: public ReplacementStrat
{
	std::vector<size_t> operator()(Matrix& features)
	{
		std::vector<size_t> attrsToUnknownEnums;
		for(size_t i = 0; i < features.cols(); i++)
			attrsToUnknownEnums.push_back(features.mostCommonValue(i));

		for(size_t i = 0; i < features.rows(); i++)
		{
			for(size_t j = 0; j < features.cols(); j++)
			{
				if(features[i][j] == UNKNOWN_VALUE)
					features[i][j] = attrsToUnknownEnums[j];
			}
		}

		return attrsToUnknownEnums;
	}

	virtual void operator()(std::vector<size_t>& attrToMissingReplacement, std::vector<double>& features)
	{
		ReplacementStrat::operator ()(attrToMissingReplacement, features);
	}
};

/**
 * Replaces the missing values with the mode 
 * @param features
 */
template <class T>
void replaceMissing(Matrix& features)
{
	T replacementStrategy;
	replacementStrategy(features);
}

#endif