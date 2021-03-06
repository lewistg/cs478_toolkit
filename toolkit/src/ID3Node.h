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
	void induceTree(Matrix& features, Matrix& labels, size_t level, std::vector<bool> excludedFeatures);

	/**
	 * Induces the tree using the Laplacian
     */
	void induceTreeByLaplacian(Matrix& features, Matrix& labels, size_t level, std::vector<bool> excludedFeatures);

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
     * Gets the number of nodes rooted at this node
     * @return 
     */
    size_t getNumDescendants();

    /**
     * Gets the maximum depth of tree rooted at this node. 
     * @param i
     * @return 
     */
    size_t getMaxDepth();

	/**
	 * Gets the indexed child node
	 */
	const ID3Node& getChildNode(size_t i) const;

	/**
	 * Indicates whether or not the node is a leaf 
     */
	bool isLeaf();

	/**
	 * Collapses this node
	 */
	void setCollapsed(bool collapsed);

	/**
	 * Appends the children nodes to end of this list
	 */
	void getChildrenNodes(std::vector<ID3Node*>& children);

private:
	/**The index of the feature that this node splits on or if
	 * this node is a leaf the label that it assigns to features that reach it*/
	long _targetAttr;

   	/**The name of the attribute that we split on*/ 
    std::string _targetAttrName;

	/**Map from an attribute value to a child node*/
	std::vector<ID3Node> _attrToChildNode; 

	/**Map from an attribute value to an attribute value name*/
	std::vector<std::string> _attrToValueName;

	/**The label to assign if this a leaf node*/
	double _labelToAssignAsLeaf;

	/**The label to assign name*/
	std::string _labelToAssignAsLeafName;

	/**The entropy during training*/
	double _trainingEntropy;

	/**Indicates whether or not the node is collapsed*/
	bool _collapsed;

    /**
     * Calculates the Laplacian for a particular attribute 
     */
    double laplacian(Matrix& features, Matrix& labels, size_t attrIndex, size_t level);

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
	 * Utility function for splitting labels by a particular attribute value
     */
	void splitOnAttr(size_t attrIndex, Matrix& features, Matrix& labels, 
			std::map<long, std::vector<long> >& attrValueBucket);

	/**
	 * Utility function for getting the most common attribute value
     */
	size_t getMajorityLabelCount(std::vector<long>& labels);

	/**
	 * Splits and induces the rest of the trees
     */
	void splitAndInduce(size_t bestAttr, Matrix& features, Matrix& labels, 
			size_t level, std::vector<bool> excludedFeatures);

	/**
	 * Splits instances according to attribute value
	 */
	void split(Matrix& features, Matrix& labels, std::vector<Matrix>& featureMatBucket, 
		std::vector<Matrix>& labelMatBucket, size_t attrIndex);
};

#endif