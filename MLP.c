//
// Created by Hwiyong on 2019-07-23.
//

#include "MLP.h"

MLP *createMLP(int layerCount, int *neuralCount, double initialLearningRate) {
    MLP *mlp = (MLP *) malloc(sizeof(MLP));
    mlp->layerCount = layerCount;
    mlp->neuralCount = neuralCount;
    mlp->learningRate = initialLearningRate;
    INITIAL_LEARNING_RATE = initialLearningRate;

    initNeuralNetwork(mlp);
    srand(time(NULL));
    randomTheta(mlp);
    randomWeight(mlp);
    return mlp;
}

void deleteMLP(MLP *mlp) {
    int layerIndex, perceptronIndex;
    for (layerIndex = 0; layerIndex < mlp->layerCount - 1; ++layerIndex) {    // hidden layer ~ output layer
        for (perceptronIndex = 0; perceptronIndex < mlp->neuralCount[layerIndex + 1]; ++perceptronIndex) {
            deletePerceptron(&mlp->neuralNetwork[layerIndex][perceptronIndex][0]);
            free(mlp->neuralNetwork[layerIndex][perceptronIndex]);
        }
        free(mlp->neuralNetwork[layerIndex]);
    }
    free(mlp->neuralCount);
    free(mlp->neuralNetwork);
    free(mlp);
}

void initNeuralNetwork(MLP *mlp) {
    int layerIndex, perceptronIndex;
    mlp->neuralNetwork = (Perceptron ***) malloc(
            sizeof(Perceptron **) * (mlp->layerCount - 1));        // except input layer
    for (layerIndex = 1; layerIndex < mlp->layerCount; ++layerIndex) {
        mlp->neuralNetwork[layerIndex - 1] = (Perceptron **) malloc(
                sizeof(Perceptron *) * mlp->neuralCount[layerIndex]);
        for (perceptronIndex = 0; perceptronIndex < mlp->neuralCount[layerIndex]; ++perceptronIndex) {
            mlp->neuralNetwork[layerIndex - 1][perceptronIndex] = (Perceptron *) malloc(sizeof(Perceptron));
            mlp->neuralNetwork[layerIndex - 1][perceptronIndex][0] = *createPerceptron(
                    mlp->neuralCount[layerIndex - 1]);
        }
    }    // mlp.neuralNetwork[0...layerCount-1] => index:[0...layerCount-2] = hidden layer, index:[layerCount-1] = output layer
}

void randomTheta(MLP *mlp) {
    int layerIndex, perceptronIndex;
    for (layerIndex = 0; layerIndex < mlp->layerCount - 1; ++layerIndex)
        for (perceptronIndex = 0; perceptronIndex < mlp->neuralCount[layerIndex + 1]; ++perceptronIndex)
            mlp->neuralNetwork[layerIndex][perceptronIndex][0].threshold = getDoubleTypeRandom(
                    -RANDOM_RANGE_ABSOLUTE_VALUE, RANDOM_RANGE_ABSOLUTE_VALUE);
}

void randomWeight(MLP *mlp) {
    int layerIndex, perceptronIndex, weightIndex;
    for (layerIndex = 0; layerIndex < mlp->layerCount - 1; ++layerIndex)
        for (perceptronIndex = 0; perceptronIndex < mlp->neuralCount[layerIndex + 1]; ++perceptronIndex)
            for (weightIndex = 0; weightIndex < mlp->neuralCount[layerIndex]; ++weightIndex)
                mlp->neuralNetwork[layerIndex][perceptronIndex][0].weight[weightIndex] = getDoubleTypeRandom(
                        -RANDOM_RANGE_ABSOLUTE_VALUE, RANDOM_RANGE_ABSOLUTE_VALUE);
}

double getDoubleTypeRandom(double startValue, double endValue) {
    // (double)rand() / RAND_MAX = 0.0 ~ 1.0
    return (double) rand() / RAND_MAX * (endValue - startValue) + startValue;
}

double ***train(MLP *mlp, int trainCount, int trainDataCount, double **trainInputDataList, int *trainAnswerIndexList) {
    double ***accuracyEpochList = (double ***) malloc(sizeof(double **) * trainCount);
    int trainIndex;
    for (trainIndex = 1; trainIndex <= trainCount; ++trainIndex)
        accuracyEpochList[trainIndex-1] = epoch(mlp, trainDataCount, trainInputDataList, trainAnswerIndexList, true);
    return accuracyEpochList;
}

// one epoch train
double **epoch(MLP *mlp, int dataCount, double **inputDataList, int *answerIndexList, bool isTrain) {
    double **accuracyList;
    int **correctList = (int **) malloc(sizeof(int *) * (mlp->neuralCount[mlp->layerCount - 1]));
    int outputLayerIndex, dataIndex;
    for (outputLayerIndex = 0; outputLayerIndex < mlp->neuralCount[mlp->layerCount - 1]; ++outputLayerIndex) {
        correctList[outputLayerIndex] = (int *) malloc(sizeof(int) * 2);
        memset(correctList[outputLayerIndex], 0, sizeof(int) * 2);
    }
    for (dataIndex = 0; dataIndex < dataCount; ++dataIndex) {
        double **outputList = forwardPropagation(mlp, inputDataList[dataIndex]);
        ++correctList[answerIndexList[dataIndex]][1];
        if (answerIndexList[dataIndex] ==
            getMaxIndex(mlp->neuralCount[mlp->layerCount - 1], outputList[mlp->layerCount - 1]))
            ++correctList[answerIndexList[dataIndex]][0];
        if (isTrain) {
            backPropagation(mlp, outputList, answerIndexList[dataIndex]);
        }
    }
    accuracyList = getAccuracyList(mlp, correctList);
    learningRateUpdate(mlp, 1 - accuracyList[mlp->neuralCount[mlp->layerCount - 1]][2]);        // update learning rate
    return accuracyList;
}

// one of data list train
double **forwardPropagation(MLP *mlp, double *inputData) {
    double sum, **output = (double **) malloc(sizeof(double *) * mlp->layerCount);
    int layerIndex, perceptronIndex, weightIndex;
    for (layerIndex = 0; layerIndex < mlp->layerCount; ++layerIndex) {
        output[layerIndex] = (double *) malloc(sizeof(double) * mlp->neuralCount[layerIndex]);
        memset(output[layerIndex], 0.0, sizeof(double) * mlp->neuralCount[layerIndex]);
        for (perceptronIndex = 0; perceptronIndex < mlp->neuralCount[layerIndex]; ++perceptronIndex) {
            if (layerIndex == 0)                     // input layer
                output[layerIndex][perceptronIndex] = inputData[perceptronIndex];
            else {
                sum = -mlp->neuralNetwork[layerIndex - 1][perceptronIndex][0].threshold;
                for (weightIndex = 0; weightIndex < mlp->neuralCount[layerIndex - 1]; ++weightIndex) {
                    if (layerIndex == 1)        // First hidden layer : sum += w * x(input layer)
                        sum += mlp->neuralNetwork[layerIndex - 1][perceptronIndex][0].weight[weightIndex] *
                               inputData[weightIndex];
                    else                        // Second hidden layer ~ Output layer : sum += w * x(hidden layer)
                        sum += mlp->neuralNetwork[layerIndex - 1][perceptronIndex][0].weight[weightIndex] *
                               output[layerIndex - 1][weightIndex];
                }
                output[layerIndex][perceptronIndex] = sigmoid(sum);
            }
        }
    }
    return output;
}

void backPropagation(MLP *mlp, double **outputList, int answerIndex) {
    // percteptron에 [0]을 붙일 것
    double **errorGradient = (double **) malloc(sizeof(double *) * (mlp->layerCount - 1));
    double output, error, hiddenLayerErrorSum;
    int layerIndex, perceptronIndex, weightIndex, nextLayerPerceptronIndex;
    for (layerIndex = mlp->layerCount - 1; layerIndex > 0; --layerIndex) {
        errorGradient[layerIndex - 1] = (double *) malloc(sizeof(double) * mlp->neuralCount[layerIndex]);
        memset(errorGradient[layerIndex - 1], 0.0, sizeof(double) * mlp->neuralCount[layerIndex]);
        for (perceptronIndex = 0; perceptronIndex < mlp->neuralCount[layerIndex]; ++perceptronIndex) {
            output = outputList[layerIndex][perceptronIndex];
            if (layerIndex == mlp->layerCount - 1) {
                error = (perceptronIndex == answerIndex ? 1 : 0) - output;
                errorGradient[layerIndex - 1][perceptronIndex] = error * sigmoidPartialDerivative(output);
                for (weightIndex = 0; weightIndex < mlp->neuralCount[layerIndex - 1]; ++weightIndex)
                    mlp->neuralNetwork[layerIndex - 1][perceptronIndex][0].weight[weightIndex] +=
                            mlp->learningRate * outputList[layerIndex - 1][weightIndex] *
                            errorGradient[layerIndex - 1][perceptronIndex];
                mlp->neuralNetwork[layerIndex - 1][perceptronIndex][0].threshold +=
                        mlp->learningRate * -1 * errorGradient[layerIndex - 1][perceptronIndex];
            } else {
                hiddenLayerErrorSum = 0.0;
                for (nextLayerPerceptronIndex = 0;
                     nextLayerPerceptronIndex < mlp->neuralCount[layerIndex + 1]; ++nextLayerPerceptronIndex)
                    hiddenLayerErrorSum += errorGradient[layerIndex][nextLayerPerceptronIndex] *
                                           mlp->neuralNetwork[layerIndex][nextLayerPerceptronIndex][0].weight[perceptronIndex];
                errorGradient[layerIndex - 1][perceptronIndex] = sigmoidPartialDerivative(output) * hiddenLayerErrorSum;
                for (weightIndex = 0; weightIndex < mlp->neuralCount[layerIndex - 1]; ++weightIndex)
                    mlp->neuralNetwork[layerIndex - 1][perceptronIndex][0].weight[weightIndex] +=
                            mlp->learningRate * outputList[layerIndex - 1][weightIndex] *
                            errorGradient[layerIndex - 1][perceptronIndex];
                mlp->neuralNetwork[layerIndex - 1][perceptronIndex][0].threshold +=
                        mlp->learningRate * -1 * errorGradient[layerIndex - 1][perceptronIndex];



            }
        }
    }
}

double **getAccuracyList(MLP *mlp, int **correctList) {
    double **accuracyList = (double **) malloc(sizeof(double *) * (mlp->neuralCount[mlp->layerCount - 1] + 1));
    int outputIndex;
    accuracyList[mlp->neuralCount[mlp->layerCount - 1]] = (double *) malloc(sizeof(double) * 3);
    memset(accuracyList[mlp->neuralCount[mlp->layerCount - 1]], 3.0, sizeof(double) * 3);
    for (outputIndex = 0; outputIndex < mlp->neuralCount[mlp->layerCount - 1]; ++outputIndex) {
        accuracyList[outputIndex] = (double *) malloc(sizeof(double) * 3);
        accuracyList[outputIndex][0] = (double) correctList[outputIndex][0];
        accuracyList[outputIndex][1] = (double) correctList[outputIndex][1];
        accuracyList[outputIndex][2] = accuracyList[outputIndex][0] / accuracyList[outputIndex][1];
        accuracyList[mlp->neuralCount[mlp->layerCount - 1]][0] += accuracyList[outputIndex][0];
        accuracyList[mlp->neuralCount[mlp->layerCount - 1]][1] += accuracyList[outputIndex][1];
    }
    accuracyList[mlp->neuralCount[mlp->layerCount - 1]][2] =
            (double) accuracyList[mlp->neuralCount[mlp->layerCount - 1]][0] /
            accuracyList[mlp->neuralCount[mlp->layerCount - 1]][1];
    return accuracyList;
}

void learningRateUpdate(MLP *mlp, double errorRate) {
    mlp->learningRate = INITIAL_LEARNING_RATE * errorRate;
}

double **test(MLP *mlp, int testDataCount, double **testInputDataList, int *testAnswerIndexList) {
    return epoch(mlp, testDataCount, testInputDataList, testAnswerIndexList, false);
}

int getMaxIndex(int arraySize, double *array) {
    int maxValueIndex = 0, valueIndex;
    for (valueIndex = 0; valueIndex < arraySize; ++valueIndex)
        if (array[valueIndex] > array[maxValueIndex])
            maxValueIndex = valueIndex;
    return maxValueIndex;
}

double sigmoid(double x) {
    return 1 / (1 + exp(-x));
}

double sigmoidPartialDerivative(double y) {
    return y * (1 - y);
}