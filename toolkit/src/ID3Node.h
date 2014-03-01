#ifndef _ID3NODE_H_
#define _ID3NODE_H_

#include <map>
#include <vector>
#include "matrix.h"

/**
 * A node in the decision tree 
 * @return 
 */
class ID3Node
{
public:
	/**
	 * Induces a tree using the given features
     * @param features
     * @param labels
     */
	void induceTree(Matrix& features, Matrix& labels);

	/**
	 * Classifies the given instance.
	 * @return The label
	 */
	double classify(const std::vector<double>& features);

private:
	/**The index of the feature that this node splits on or if
	 * this node is a leaf the label that it assigns to features that reach it*/
	long _attrToSplitOrAssign;

	/**Map from an attribute value to a child node*/
	std::map<long, ID3Node> _attrToNode; 

	/**
	 * Calculates information gain by splitting on the given
	 * attribute.
     */
	double infoGain(Matrix& features, Matrix& labels, size_t attrIndex);

	/**
	 * Calculates the information in the current set of features
	 */
	double info(Matrix& labels);
};

#endif