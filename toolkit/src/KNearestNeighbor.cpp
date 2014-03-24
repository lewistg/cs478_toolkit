#include <utility>
#include <ctime>
#include <cassert>
#include <cmath>
#include <math.h>
#include <iostream>
#include <map>
#include "KNearestNeighbor.h"
#include "data_utils.h"

#define WEIGHTS false

KNearestNeighbor::KNearestNeighbor():_k(5)
{

}

void KNearestNeighbor::train(Matrix& features, Matrix& labels)
{
	// non-reduction
	_examples = Matrix(features);
	_examples.copyPart(features, 0, 0, features.rows(), features.cols());

	_exampleLabels = Matrix(labels);
	_exampleLabels.copyPart(labels, 0, 0, labels.rows(), labels.cols());

	for(size_t col = 0; col < _examples.cols(); col++)
	{
		_colMaxes.push_back(_examples.columnMax(col));
		_colMins.push_back(_examples.columnMin(col));
	}

	for(size_t row = 0; row < _examples.rows(); row++)
		normalizeFeatures(_examples.row(row));

	_minLabel = labels.columnMin(0);
	_maxLabel = labels.columnMax(0);

	//leaveOneOutReduction();
	//growthReduction();
}

void KNearestNeighbor::predictWithIgnore(const std::vector<double>& features, std::vector<double>& labels, 
	std::vector<bool>* ignoredExamples)
{
	if(ignoredExamples != NULL)
		assert(ignoredExamples->size() == _examples.rows());

	std::vector<double> normFeatures(features);
	normalizeFeatures(normFeatures);

	std::priority_queue<std::pair<size_t, double>, std::vector<std::pair<size_t, double> >, PairCmpr> 
	nearestKInstances = 
            std::priority_queue<std::pair<size_t, double>, std::vector<std::pair<size_t, double> >, PairCmpr> ();
	for(size_t i = 0; i < _examples.rows(); i++)
	{
		if(ignoredExamples != NULL && ignoredExamples->at(i))
			continue;

		double exampleDist = dist(_examples.row(i), normFeatures);
		//double exampleDist = manhattanDist(_examples.row(i), normFeatures);
		if(nearestKInstances.size() >= _k)
		{
			if(nearestKInstances.top().second > exampleDist)
			{
				nearestKInstances.push(std::pair<size_t, double>(i, exampleDist));
				if(nearestKInstances.size() > _k)
					nearestKInstances.pop(); 
			}
		}
		else
		{
				nearestKInstances.push(std::pair<size_t, double>(i, exampleDist));
		}

	}
	assert(nearestKInstances.size() == _k);
	
    //std::cout << "Nearest using Manhattan neighbors: " << std::endl;
	//printQueue(nearestKInstances);

    bool isContinuous = (_exampleLabels.valueCount(0) == 0);
    if(isContinuous)
		regressionPrediction(normFeatures, labels, nearestKInstances);
    else
		predictNominal(normFeatures, labels, nearestKInstances);
}

void KNearestNeighbor::predict(const std::vector<double>& features, std::vector<double>& labels)
{
	predictWithIgnore(features, labels);
}

void KNearestNeighbor::predictNominal(const std::vector<double>& normFeatures, std::vector<double>& labels, 
	std::priority_queue<std::pair<size_t, double>, std::vector<std::pair<size_t, double> >, PairCmpr>&
	nearestKInstances)
{
	// take a vote of the closest neighbors
	std::map<double, size_t> labelCounts;
	while(!nearestKInstances.empty())
	{
		size_t nearNeighborIndex = nearestKInstances.top().first;
		nearestKInstances.pop();

		double nearNeighborLabel = _exampleLabels.row(nearNeighborIndex)[0];
		double nearNeighborDist = dist(_examples.row(nearNeighborIndex), normFeatures);
		if(nearNeighborDist <= 0.00000001)
		{
			labels[0] = nearNeighborLabel;
			return;
		}

        // weighted voting
		if(WEIGHTS)
		{
			double weight = 1.0 / (nearNeighborDist * nearNeighborDist);
			labelCounts[nearNeighborLabel] += weight * 1;
		}
		else
		{
			labelCounts[nearNeighborLabel] += 1;
		}
	}

	size_t maxCount = 0;
	double maxLabel = 0.0;
	for(std::map<double, size_t>::iterator itr = labelCounts.begin(); itr != labelCounts.end(); itr++)
	{
		if(itr->second > maxCount)
		{
			maxCount = itr->second;
			maxLabel = itr->first;
		}
	}
	
	labels[0] = maxLabel;
}

void KNearestNeighbor::regressionPrediction(const std::vector<double>& normFeatures, std::vector<double>& labels, 
	std::priority_queue<std::pair<size_t, double>, std::vector<std::pair<size_t, double> >, PairCmpr>&
	nearestKInstances)
{
	double labelSum = 0.0;
	double weightSum = 0.0;
	while(!nearestKInstances.empty())
	{
		size_t nearNeighborIndex = nearestKInstances.top().first;
		nearestKInstances.pop();

		double nearNeighborLabel = _exampleLabels.row(nearNeighborIndex)[0];

		// test
		//nearNeighborLabel = (nearNeighborLabel - _minLabel) / (_maxLabel - _minLabel);

		double nearNeighborDist = dist(_examples.row(nearNeighborIndex), normFeatures);
		if(nearNeighborDist <= 0.00000001)
		{
			labels[0] = nearNeighborLabel;
			return;
		}

        if(WEIGHTS)
        {
			double weight = 1.0 / (nearNeighborDist * nearNeighborDist);
			weightSum += weight;
			labelSum += weight * nearNeighborLabel;
        }
        else
        {
			labelSum += nearNeighborLabel; 
        }
	}

	if(WEIGHTS)
	{
		double avgLabel = labelSum / weightSum; 
		labels[0] = avgLabel;
	}
	else
	{
		double avgLabel = labelSum / static_cast<double>(_k);
		labels[0] = avgLabel;
	}
}

double KNearestNeighbor::dist(const std::vector<double>& features, const std::vector<double>& example)
{
	//return manhattanDist(features, example);
	return heom(features, example);

	/*assert(features.size() == example.size());
	
	double delta = 0;
	double deltaSum = 0;
	for(size_t i = 0; i < features.size(); i++)
	{
		delta = features[i]	- example[i];
		deltaSum += (delta * delta);
	}

	double dist = sqrt(deltaSum);
	return dist;*/
}

double KNearestNeighbor::heom(const std::vector<double>& features, const std::vector<double>& example)
{
	assert(_examples.cols() == features.size());
	assert(_examples.cols() == example.size());

	double sum = 0.0;
	for(size_t i = 0; i < _examples.cols(); i++)
	{
		if(features[i] == UNKNOWN_VALUE || example[i] == UNKNOWN_VALUE)
		{
			sum += 1.0;
		}
		else if(_examples.valueCount(0) == 0)
		{
			double delta = (features[i] - example[i]);
			sum += delta * delta;
		}
		else if(features[i] != example[i]) 
		{
			sum += 1.0;
		}
	}

	double dist = sqrt(sum);
	return dist;
}

double KNearestNeighbor::manhattanDist(const std::vector<double>& features, const std::vector<double>& example)
{
	double dist = 0.0;
	for(size_t i = 0; i < features.size(); i++)
		dist += fabs(features[i] - example[i]);

	return dist;
}

void KNearestNeighbor::normalizeFeatures(std::vector<double>& features)
{
	for(size_t col = 0; col < _examples.cols(); col++)
	{
		double delta = _colMaxes[col] - _colMins[col];
		features[col] = (features[col] - _colMins[col]) / delta;
	}
}

void KNearestNeighbor::leaveOneOutReduction()
{
    std::vector<bool> ignoredExample(_examples.rows(), false);
	std::vector<double> label(1);

    // create a validation set
	/*Matrix trainingSet; 
	Matrix trainingSetLabels;
	Matrix validationSet; 
	Matrix validationSetLabels;

	void partitionTrainAndVal(Matrix& features, Matrix& labels, 
			trainingSet, trainingSetLabels,
			validationSet, validationSetLabels);*/

    // reduce to 2000 
	/*Rand r(time(NULL));
	_examples.shuffleRows(r, &_exampleLabels);
	for(size_t i = 2000; i < _examples.rows(); i++)
		ignoredExample[i] = true;

	size_t nRows =  2000;
	if(_examples.rows() < nRows)
		nRows = _examples.rows();*/

	size_t nRows = _examples.rows();
    for(size_t i = 0; i < nRows; i++)
    {
        ignoredExample[i] = true;
		predictWithIgnore(_examples.row(i), label, &ignoredExample);

        bool correctlyPredicted = false; 
		if(_exampleLabels.numAttrValues(0) == 0)
			correctlyPredicted = (fabs(label[0] - _exampleLabels[i][0]) < 0.1);
		else
			correctlyPredicted = (label[0] == _exampleLabels[i][0]);

        if(!correctlyPredicted)
			ignoredExample[i] = false;
    }

	Matrix reducedExamples(_examples);
    Matrix reducedExampleLabels(_exampleLabels);

    for(size_t i = 0; i < _examples.rows(); i++)
    {
        if(!ignoredExample[i])
        {
            reducedExamples.copyRow(_examples.row(i));
            reducedExampleLabels.copyRow(_exampleLabels.row(i));
        }
    }

	_examples = Matrix(reducedExamples);
	_examples.copyPart(reducedExamples, 0, 0, reducedExamples.rows(), reducedExamples.cols());

	_exampleLabels = Matrix(_exampleLabels);
	_exampleLabels.copyPart(reducedExampleLabels, 0, 0, reducedExampleLabels.rows(), reducedExampleLabels.cols());

	std::cout << "Reduced to: " << _examples.rows() << std::endl;
}


void KNearestNeighbor::growthReduction()
{
    std::vector<bool> ignoredExample(_examples.rows(), true);
	std::vector<double> label(1);

	Rand r(time(NULL));
	_examples.shuffleRows(r, &_exampleLabels);

	// keep the first k instances
	for(size_t i = 0; i < _k; i++)
		ignoredExample[i] = false;

    for(size_t i = _k; i < _examples.rows(); i++)
    {
		predictWithIgnore(_examples.row(i), label, &ignoredExample);
        bool correctlyPredicted = false; 
		if(_exampleLabels.numAttrValues(0) == 0)
			correctlyPredicted = (fabs(label[0] - _exampleLabels[i][0]) < 0.1);
		else
			correctlyPredicted = (label[0] == _exampleLabels[i][0]);

        if(!correctlyPredicted)
			ignoredExample[i] = false;
	}

	Matrix reducedExamples(_examples);
    Matrix reducedExampleLabels(_exampleLabels);
    for(size_t i = 0; i < _examples.rows(); i++)
    {
        if(!ignoredExample[i])
        {
            reducedExamples.copyRow(_examples.row(i));
            reducedExampleLabels.copyRow(_exampleLabels.row(i));
        }
    }

	_examples = Matrix(reducedExamples);
	_examples.copyPart(reducedExamples, 0, 0, reducedExamples.rows(), reducedExamples.cols());

	_exampleLabels = Matrix(_exampleLabels);
	_exampleLabels.copyPart(reducedExampleLabels, 0, 0, reducedExampleLabels.rows(), reducedExampleLabels.cols());

	std::cout << "Reduced to: " << _examples.rows() << std::endl;
}

void KNearestNeighbor::printQueue(std::priority_queue<std::pair<size_t, double>, 
	std::vector<std::pair<size_t, double> >, PairCmpr>& q)
{
	std::vector<std::pair<size_t, double> > buffer; 

	std::cout << "Queue: " << std::endl;
	while(!q.empty())
	{
		std::cout << q.top().first << " - " << q.top().second << std::endl;
		std::cout << vectorToString(_examples[q.top().first]) << std::endl;
		buffer.push_back(q.top());
		q.pop();
	}

	for(size_t i = 0; i < buffer.size(); i++)
	{
		q.push(buffer[i]);
	}
}

bool KNearestNeighbor::equalQueues(std::priority_queue<std::pair<size_t, double>, 
	std::vector<std::pair<size_t, double> >, PairCmpr>& q1,

	std::priority_queue<std::pair<size_t, double>, 
	std::vector<std::pair<size_t, double> >, PairCmpr>& q2)
{

	std::vector<std::pair<size_t, double> > buffer1; 
	std::vector<std::pair<size_t, double> > buffer2; 

	bool equal = true;
	assert(q1.size() == q2.size());
	while(!q1.empty())
	{
		if(q1.top().first != q2.top().first)
		{
			equal = false;	
			break;
		}

		buffer1.push_back(q1.top());
		q1.pop();

		buffer2.push_back(q2.top());
		q2.pop();
	}

	for(size_t i = 0; i < buffer1.size(); i++)
	{
		q1.push(buffer1[i]);
		q2.push(buffer2[i]);
	}
	
	return equal;
}