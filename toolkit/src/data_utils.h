#ifndef _DATA_UTILS_H_
#define _DATA_UTILS_H_

#include <string>
#include <sstream>
#include <vector>

template <class T>
std::string vectorToString(const std::vector<T>& array)
{
	std::stringstream ss;
	for(size_t i = 0; i < array.size(); i++)
	{
		ss << array[i];
		if(i < array.size() - 1)
			ss << ", ";
	}

	return ss.str();
}

#endif