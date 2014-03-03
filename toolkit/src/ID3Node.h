#ifndef _ID3NODE_H_
#define _ID3NODE_H_

#include <map>
#include <vector>
#include <utility>
#include <string>
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

	/**
	 * Gets the target attribute (the one that it splits on)
	 */
	long getTargetAttr() const;

	/**
	 * Gets the target attribute name
	 */
	std::string getTargetAttrName() const;

	/**
	 * Target attribute value
     */
	std::string getTargetAttrValue(size_t i) const;

	/**
	 * Gets the label to assign (if there is one).
	 * If this node does not assign a label, -1 is returned
	 */
	long getLabelToAssign() const;

	/**
	 * Gets the name of the attribute value to be assigned
	 */
	std::string getLabelToAssignName() const;

	/**
	 * Get number of child nodes
	 */
	size_t getNumChildNodes() const;

	/**
	 * Gets the indexed child node
	 */
	const ID3Node& getChildNode(size_t i) const;

private:
	/**The log*/
	static ID3Logger _log;

	/**The index of the feature that this node splits on or if
	 * this node is a leaf the label that it assigns to features that reach it*/
	long _targetAttr;

   	/**The name of the attribute that we split on*/ 
    std::string _targetAttrName;

	/**Map from an attribute value to a child node*/
	std::vector<ID3Node> _attrToNode; 

	/**Map from an attribute value to an attribute value name*/
	std::vector<std::string> _attrToValueName;

	/**The label to assign if this a leaf node*/
	double _labelToAssign;

	/**The label to assign name*/
	std::string _labelToAssignName;

	/**The entropy during training*/
	double _trainingEntropy;

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