#include <cassert>
#include <ctime>
#include <iostream>
#include "ID3.h"
#include "ID3TreePlotter.h"

/*template <class T>
ID3<T>::ID3()
{

}

template <class T>
ID3<T>::~ID3()
{

}

template <class T>
void ID3<T>::train(Matrix& features, Matrix& labels)
{
	// create a validation set
	Rand r(time(NULL));
	features.shuffleRows(r, &labels);

	double percentValidation = 0.25;
	size_t validationSetSize = static_cast<size_t>(std::max(percentValidation * features.rows(), 1.0));
	size_t trainingSetSize = features.rows() - validationSetSize;
	assert(validationSetSize > 0 && validationSetSize < features.rows());

	Matrix trainingSet;
	trainingSet.copyPart(features, 0, 0, trainingSetSize, features.cols());

	Matrix trainingSetLabels;
	trainingSetLabels.copyPart(labels, 0, 0, trainingSetSize, labels.cols());

	Matrix validationSet;
	validationSet.copyPart(features, trainingSetSize, 0, validationSetSize, features.cols());

	Matrix validationSetLabels;
	validationSetLabels.copyPart(labels, trainingSetSize, 0, validationSetSize, labels.cols());

	//_root.induceTree(features, labels, 0);
	std::vector<bool> excludedFeatures(trainingSet.cols(), false);
	_root.induceTree(trainingSet, trainingSetLabels, 0, excludedFeatures);
	ID3TreePlot::plotTree("before_prune", _root);

	pruneTree(validationSet, validationSetLabels);
	ID3TreePlot::plotTree("after_prune", _root);
}

template <class T>
void ID3<T>::predict(const std::vector<double>& features, std::vector<double>& labels)
{
	labels[0] = _root.classify(features);
} 

template <class T>
void ID3<T>::pruneTree(Matrix& validationSet, Matrix& validationSetLabels)
{
	// prune until accuracy stops improving
	std::vector<ID3Node*> treeNodes;
	treeNodes.push_back(&_root);
	_root.getChildrenNodes(treeNodes);
	double prevAcc = calcValSetAcc(validationSet, validationSetLabels);
	double bestAfterPruneAcc = 0.0;
	size_t nPruned = 0;
	//std::cout << "Number of nodes in tree: " << treeNodes.size() << std::endl;
	//std::cout << "Prev acc: " << prevAcc << std::endl;
	while(true)
	{
		bestAfterPruneAcc = 0.0;
		size_t bestNodeToPrune = 0;
		for(size_t i = 0; i < treeNodes.size(); i++)
		{
			if(!treeNodes[i]->isLeaf())
			{
				treeNodes[i]->setCollapsed(true);
				double acc = calcValSetAcc(validationSet, validationSetLabels);
				if(acc > bestAfterPruneAcc)
				{
					bestNodeToPrune = i;
					bestAfterPruneAcc = acc;
				}
				treeNodes[i]->setCollapsed(false);
			}
		}
		
		if(bestAfterPruneAcc > prevAcc)
		{
			treeNodes[bestNodeToPrune]->setCollapsed(true);
			prevAcc = bestAfterPruneAcc;
			nPruned += 1;
		}
		else
		{
			break;
		}
	}
	std::cout << "Number pruned: " << nPruned << std::endl;
	ID3::id3Log.logPrunedNodes(nPruned);
}

template <class T>
double ID3<T>::calcValSetAcc(Matrix& validationSet, Matrix& validationSetLabels)
{
	double nRight = 0;
	double total = validationSet.rows();
	std::vector<double> prediction(1);
	for(size_t i = 0; i < validationSet.rows(); i++)
	{
		double actualLabel = validationSetLabels.row(i)[0];
		predict(validationSet.row(i), prediction);

		if(actualLabel == prediction[0])
			nRight += 1;
	}

	double percentRight = nRight / total;
	return percentRight;
}*/