//
// Created by Hwiyong on 2019-07-23.
//

#include <stdio.h>
#include "Perceptron.h"

Perceptron *createPerceptron(int weightCount) {
    Perceptron *p = (Perceptron *) malloc(sizeof(Perceptron));
    p->threshold = 0.0;
    p->weight = (double *) malloc(sizeof(double) * weightCount);
    memset(p->weight, 0.0, sizeof(double) * weightCount);
    return p;
}

void deletePerceptron(Perceptron *p) {
    free(p->weight);
}