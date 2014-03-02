#include "ID3.h"
#include "ID3TreePlotter.h"

ID3::ID3()
{

}

ID3::~ID3()
{

}

void ID3::train(Matrix& features, Matrix& labels)
{
	_root.induceTree(features, labels, 0);
	ID3TreePlot::plotTree(_root);
}

void ID3::predict(const std::vector<double>& features, std::vector<double>& labels)
{
} 