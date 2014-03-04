#include <cassert>
#include <cmath>
#include <iostream>
#include "ID3Node.h"
#include "data_utils.h"

ID3Logger ID3Node::_log;

ID3Node::ID3Node():_targetAttr(-1), _labelToAssign(0.0)
{

}

void ID3Node::setLabelToAssign(double labelToAssign) 
{
	_labelToAssign = labelToAssign;
}

long ID3Node::getTargetAttr() const
{
	return _targetAttr; 
}

std::string ID3Node::getTargetAttrName() const
{
	return _targetAttrName;
}

std::string ID3Node::getTargetAttrValue(size_t i) const
{
	return _attrToValueName[i];
}

long ID3Node::getLabelToAssign() const
{
	return _labelToAssign;
}

std::string ID3Node::getLabelToAssignName() const
{
	return _labelToAssignName;
}

size_t ID3Node::getNumChildNodes() const
{
	return _attrToNode.size();
}

const ID3Node& ID3Node::getChildNode(size_t i) const
{
	return _attrToNode[i];
}

void ID3Node::induceTree(Matrix& features, Matrix& labels, size_t level)
{
	assert(labels.cols() == 1);
	assert(labels.rows() > 0);

	// if the labels are homogenous 
	bool allHomog = true;
	for(size_t i = 1; i < labels.rows(); i++)
	{
		if(labels[0][0] != labels[i][0])	
		{
			allHomog = false;
			break;
		}
	}
	if(allHomog)
	{
		_labelToAssign = labels[0][0];
		_labelToAssignName = labels.attrValue(0, labels[0][0]);
		return;
	}

	double infoS = info(labels);
	_log.logNodeEntropy(infoS, level);

	// find the feature that gives the greatest information gain
	size_t bestAttr = 0;
	double maxInfoGain = 0.0;
	for(size_t i = 0; i < features.cols(); i++)
	{
		double gain = infoGain(features, labels, i, infoS, level);
		if(gain > maxInfoGain)
		{
			maxInfoGain = gain;
			bestAttr = i;
		}
	}

	bool noInfoGain = maxInfoGain < 0.000001;
	if(noInfoGain)
	{
		_labelToAssign = labels.mostCommonValue(0);
		_labelToAssignName = labels.attrValue(0, _labelToAssign);
		return;
	}

	_targetAttr = bestAttr;
	_targetAttrName = features.attrName(_targetAttr);

	// split and induce the child trees
	std::vector<Matrix> featureMatBucket;
	std::vector<Matrix> labelMatBucket;
	split(features, labels, featureMatBucket, labelMatBucket, bestAttr);

	_attrToNode.resize(featureMatBucket.size());
	_attrToValueName.resize(featureMatBucket.size());

	for(size_t i = 0; i < featureMatBucket.size(); i++)
	{
		_attrToValueName[i] = features.attrValue(_targetAttr, i);
		if(featureMatBucket[i].rows() == 0)
		{
			assert(labelMatBucket[i].rows() == 0);
			_attrToNode[i].setLabelToAssign(labels.mostCommonValue(0));
		}
		else
		{
			_attrToNode[i].induceTree(featureMatBucket[i], labelMatBucket[i], level + 1);
		}
	}
}

double ID3Node::classify(const std::vector<double>& features)
{
	if(_targetAttr == -1)
		return _labelToAssign;
	else
		return _attrToNode[features[_targetAttr]].classify(features);
}

double ID3Node::infoGain(Matrix& features, Matrix& labels, size_t attrIndex, double infoS, size_t level)
{
	assert(attrIndex < features.cols());
	assert(labels.cols() == 1);

	std::map<long, std::vector<long> > attrValueBucket;
	for(size_t i = 0; i < features.rows(); i++)
	{
		double intPart = 0;
		assert(modf(features.row(i)[attrIndex], &intPart) == 0.0);

		long attrValue = static_cast<long>(features[i][attrIndex]);
		attrValueBucket[attrValue].push_back(labels.row(i)[0]);
	}

	
	double total = static_cast<double>(labels.rows());
	double infoAfterSplit = 0.0;
	for(std::map<long, std::vector<long> >::iterator itr = attrValueBucket.begin(); itr != attrValueBucket.end(); itr++)
	{
		double nLabelsInBucket = itr->second.size(); 	
		double bucketEntropy =  info(itr->second);

		infoAfterSplit += (nLabelsInBucket / total) * bucketEntropy;
	}

	double infoGain = infoS - infoAfterSplit;
	_log.logSplitInfoGain(attrIndex, infoGain, level);
	return infoGain; 
}

double ID3Node::info(std::vector<long> labels)
{
	if(labels.size() == 0)
		return 0;

	std::map<long, double> labelToCount;
	for(size_t i = 0; i < labels.size(); i++)
	{
		double intPart = 0;
		assert(modf(labels[i], &intPart) == 0.0);
		labelToCount[static_cast<long>(labels[i])] += 1.0;
	}

	double total = labels.size();
	double entropy = 0.0;
	for(std::map<long, double>::iterator itr = labelToCount.begin(); itr != labelToCount.end(); itr++)
		entropy += (itr->second / total) * log2(itr->second / total);

	return -entropy;
}

double ID3Node::info(Matrix& labels)
{
	assert(labels.cols() == 1);

	if(labels.rows() == 0)
		return 0;

	std::vector<long> labelsAsVector;
	for(size_t i = 0; i < labels.rows(); i++)
	{
		double intPart = 0;
		assert(modf(labels.row(i)[0], &intPart) == 0.0);
		labelsAsVector.push_back(static_cast<long>(labels.row(i)[0]));
	}

	return info(labelsAsVector);
}

void ID3Node::split(Matrix& features, Matrix& labels, std::vector<Matrix>& featureMatBucket, std::vector<Matrix>& labelMatBucket, size_t attrIndex)
{
	size_t nDiffValues = features.valueCount(attrIndex);
	assert(nDiffValues > 0);

	featureMatBucket = std::vector<Matrix>(nDiffValues, features);
	labelMatBucket = std::vector<Matrix>(nDiffValues, labels);

	for(size_t i = 0; i < features.rows(); i++)
	{
		double intPart = 0;
		assert(modf(features.row(i)[attrIndex], &intPart) == 0.0);
		featureMatBucket[static_cast<long>(features.row(i)[attrIndex])].copyRow(features.row(i));
		labelMatBucket[static_cast<long>(features.row(i)[attrIndex])].copyRow(labels.row(i));
	}
}