#include "ID3.h"

ID3::ID3()
{

}

ID3::~ID3()
{

}

void ID3::train(Matrix& features, Matrix& labels)
{
	_root.induceTree(features, labels);
}

void ID3::predict(const std::vector<double>& features, std::vector<double>& labels)
{
} 