#ifndef _HAC_H_
#define _HAC_H_

#include "learner.h"


class HAC: public SupervisedLearner
{
    public:

	/**
	 * @override
	 */
	virtual void train(Matrix& features, Matrix& labels);

	/**
	 * @override
     */
	virtual void predict(const std::vector<double>& features, std::vector<double>& labels);

	private:
};

#endif