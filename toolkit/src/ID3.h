#ifndef _ID3_H_
#define _ID3_H_

#include "learner.h"

class ID3: public SupervisedLearner
{
public:
    /**
     * Constructor
     */
    ID3();

	/**
	 * Destructor
     */
	~ID3();

	/**
	 * Override
	 */
	virtual void train(Matrix& features, Matrix& labels);

	/**
	 * Override
	 */
	virtual void predict(const std::vector<double>& features, std::vector<double>& labels);

private:
	/**The root node*/
};

#endif