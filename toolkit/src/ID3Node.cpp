#include <cassert>
#include <cmath>
#include "ID3Node.h"

void ID3Node::induceTree(Matrix& features, Matrix& labels)
{

}

double ID3Node::classify(const std::vector<double>& features)
{
	return 0;
}

double ID3Node::infoGain(Matrix& features, Matrix& labels, size_t attrIndex)
{
	// calculate the entropy of the current info 
	return 0;
}

double ID3Node::info(Matrix& labels)
{
	assert(labels.cols() == 1);

	std::map<long, double> labelToCount;
	for(size_t i = 0; i < labels.rows(); i++)
	{
		assert(modf(labels.row(i)[0], NULL) == 0.0);
		labelToCount[static_cast<long>(labels.row(i)[0])] += 1.0;
	}

	double total = labels.rows();
	double entropy = 0.0;
	for(std::map<long, double>::iterator itr = labelToCount.begin(); itr != labelToCount.end(); itr++)
		entropy = (itr->second / total) * log2(itr->second / total);

	return -entropy;
}