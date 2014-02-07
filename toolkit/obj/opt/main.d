../obj/opt/main.o: main.cpp learner.h matrix.h rand.h baseline.h error.h filter.h \
 perceptron.h perceptron_rule_perceptron_node.h perceptron_node.h \
 BackProp.h BackPropLayer.h BackPropUnit.h
	g++ -Wall -O3 -c main.cpp -o ../obj/opt/main.o
