//
// Created by Hwiyong on 2019-07-23.
//

#ifndef MLP_H
#define MLP_H

#include <stdio.h>
#include <stdbool.h>
#include <math.h>
#include <time.h>
#include "Perceptron.h"

#define RANDOM_RANGE_ABSOLUTE_VALUE 1.0

double INITIAL_LEARNING_RATE;

typedef struct {
    int layerCount;
    int *neuralCount;
    double learningRate;
    Perceptron ***neuralNetwork;
} MLP;

MLP *createMLP(int layerCount, int *neuralCount, double initialLearningRate);

void deleteMLP(MLP *mlp);

void initNeuralNetwork(MLP *mlp);

void randomTheta(MLP *mlp);

void randomWeight(MLP *mlp);

double getDoubleTypeRandom(double startValue, double endValue);

double ***train(MLP *mlp, int trainCount, int trainDataCount, double **trainInputDataList, int *trainAnswerIndexList);

double **epoch(MLP *mlp, int dataCount, double **inputDataList, int *answerIndexList, bool isTrain);

double **frontPropagation(MLP *mlp, double *inputData);

void backPropagation(MLP *mlp, double **outputList, int answerIndex);

double **getAccuracyList(MLP *mlp, int **correctList);

void learningRateUpdate(MLP *mlp, double errorRate);

double **test(MLP *mlp, int testDataCount, double **testInputDataList, int *testAnswerIndexList);

int getMaxIndex(int arraySize, double *array);

double sigmoid(double x);

double sigmoidPartialDerivative(double y);

#endif //MLP_H
