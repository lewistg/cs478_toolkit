#ifndef _ID3NODE_H_
#define _ID3NODE_H_

#include <map>
#include <vector>
#include <utility>
#include "matrix.h"
#include "ID3Logger.h"
#include "ID3TreePlotter.h"

/**
 * A node in the decision tree 
 * @return 
 */
class ID3Node
{
public:
	/**
	 * Constructors
     */
	ID3Node();

	/**
	 * Induces a tree using the given features
     * @param features
     * @param labels
     */
	void induceTree(Matrix& features, Matrix& labels, size_t level);

	/**
	 * Classifies the given instance.
	 * @return The label
	 */
	double classify(const std::vector<double>& features);

	/**
	 * Sets the label to assign
	 */
	void setLabelToAssign(double labelToAssign);

private:
	friend void ID3TreePlot::plotTree(const ID3Node& root);
	friend void ID3TreePlot::explore(const ID3Node& node, std::ofstream& treePlot);

	/**The log*/
	static ID3Logger _log;

	/**The index of the feature that this node splits on or if
	 * this node is a leaf the label that it assigns to features that reach it*/
	long _targetAttr;

	/**Map from an attribute value to a child node*/
	std::vector<ID3Node> _attrToNode; 

	/**The label to assign if this a leaf node*/
	double _labelToAssign;

	/**
	 * Calculates information gain by splitting on the given
	 * attribute.
     */
	double infoGain(Matrix& features, Matrix& labels, size_t attrIndex, double infoS, size_t level);

	/**
	 * Calculates the information in the current set of features
     */
	double info(std::vector<long> labels);

	/**
	 * Calculates the information in the current set of features
	 */
	double info(Matrix& labels);

	/**
	 * Splits instances according to attribute value
	 */
	void split(Matrix& features, Matrix& labels, std::vector<Matrix>& featureMatBucket, std::vector<Matrix>& labelMatBucket, size_t attrIndex);
};

#endif