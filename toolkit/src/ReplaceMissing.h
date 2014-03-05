#ifndef _REPLACEMISSING_H_
#define _REPLACEMISSING_H_

#include <vector>
#include "matrix.h"


/**
 * Replaces the missing values with a new attribute
 * @param features
 */
struct ReplaceWithAttribute
{
	void operator()(Matrix& features)
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