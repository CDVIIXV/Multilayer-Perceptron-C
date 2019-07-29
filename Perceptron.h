//
// Created by Hwiyong on 2019-07-23.
//

#ifndef PERCEPTRON_H
#define PERCEPTRON_H

#include <stdlib.h>
#include <string.h>

typedef struct {
    double threshold;
    double *weight;
} Perceptron;

Perceptron *createPerceptron(int weightCount);

void deletePerceptron(Perceptron *p);

#endif //PERCEPTRON_H
