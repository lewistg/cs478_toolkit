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
	/**
	 * Returns a matrix with how the data was replaced
     * @param features
     * @return 
     */
	Matrix operator()(Matrix& features)
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

		Matrix updatedMatrixEnums(features);
		return updatedMatrixEnums;
    }

	/**
	 * Replaces the missing data in a vector
	 */
	void operator()(Matrix& updatedMatrixEnums, std::vector<double>& features)
	{
		for(size_t i = 0; i < features.size(); i++)
		{
			if(features[i] == UNKNOWN_VALUE)
				features[i] = updatedMatrixEnums.attrValue(i, "unknown");
		}
	}
};

/**
 * Replaces the missing data with the most common value
 * @param features
 */
struct ReplaceWithMode
{
	void operator()(Matrix& features)
	{
		for(size_t i = 0; i < features.rows(); i++)
		{
			for(size_t j = 0; j < features.cols(); j++)
			{
				if(features[i][j] == UNKNOWN_VALUE)
					features[i][j] = features.mostCommonValue(j);
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