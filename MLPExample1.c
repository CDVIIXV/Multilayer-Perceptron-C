//
// Created by Hwiyong on 2019-07-23.
//

#include "MLP.h"

// 데이터 상수 부분
#define NUMBER_LIMIT 1000        // data로 사용할 숫자(1 ~ NUMBER_LIMIT)
#define BINARY_DIGIT_LENGTH 10    // 무조건 NUMBER_LIMIT < 2^BINARY_DIGIT_LENGTH (=입력층 개수)
#define MODULAR_COUNT 10        // 나머지 (=출력층 개수)

// MLP 상수 부분
#define INITIAL_LEARNING_RATE 0.1  // 초기 학습률
#define TRAIN_RATE 0.75     // 총 데이터에서 훈련 데이터 비중
#define TRAIN_COUNT 3000    // 훈련 횟수 (값이 너무 크면 Segment fault)
#define LAYER_COUNT 4       // 신경망 층 개수
const int NEURON_COUNT[] = {BINARY_DIGIT_LENGTH, 16, 20, MODULAR_COUNT};    // 뉴런층 개수

double **getInputDataList(int dataLength, int *data);

int *getAnswerIndexList(int dataLength, int *data);

void accuracyListPrint(int listLength, double **accuracyList);

int main(void) {
    int i, j;
    int *neuralCount = (int *) malloc(sizeof(int) * LAYER_COUNT);
    for (i = 0; i < LAYER_COUNT; ++i)
        neuralCount[i] = NEURON_COUNT[i];

    // 데이터 생성
    int *data = (int *) malloc(sizeof(int) * NUMBER_LIMIT);
    for (i = 0; i < NUMBER_LIMIT; ++i)
        data[i] = i + 1;


    // MLP 생성
    MLP *mlp = createMLP(LAYER_COUNT, neuralCount, INITIAL_LEARNING_RATE);

    // 데이터를 MLP 입력층과 출력층에 맞도록
    // 입력데이터와 정답데이터로 변환
    double **inputDataList = getInputDataList(NUMBER_LIMIT, data);
    int *answerIndexList = getAnswerIndexList(NUMBER_LIMIT, data);

     // 데이터를 섞는 부분 생략

    // [0] = 훈련용, [1] = 시험용
    double **trainInputDataList = (double **) malloc(sizeof(double *) * NUMBER_LIMIT * TRAIN_RATE);
    int *trainAnswerIndexList = (int *) malloc(sizeof(int) * NUMBER_LIMIT * TRAIN_RATE);
    for (i = 0; i < NUMBER_LIMIT * TRAIN_RATE; ++i) {
        trainInputDataList[i] = (double *) malloc(sizeof(double *) * BINARY_DIGIT_LENGTH);
        trainInputDataList[i] = inputDataList[i];
        trainAnswerIndexList[i] = answerIndexList[i];
    }
    double **testInputDataList = (double **) malloc(sizeof(double *) * NUMBER_LIMIT * (1 - TRAIN_RATE));
    int *testAnswerIndexList = (int *) malloc(sizeof(int) * NUMBER_LIMIT * (1 - TRAIN_RATE));
    for (i = 0; i < NUMBER_LIMIT * (1 - TRAIN_RATE); ++i) {
        testInputDataList[i] = (double *) malloc(sizeof(double *) * BINARY_DIGIT_LENGTH);
        testInputDataList[i] = inputDataList[(int) (NUMBER_LIMIT * TRAIN_RATE) + i];
        testAnswerIndexList[i] = answerIndexList[(int) (NUMBER_LIMIT * TRAIN_RATE) + i];
    }

    // 훈련 결과 출력
    double ***trainAccuracyList = train(mlp, TRAIN_COUNT, (int) (NUMBER_LIMIT * TRAIN_RATE), trainInputDataList,
                                        trainAnswerIndexList);
    printf("Train\n");
    for (i = 0; i < TRAIN_COUNT; ++i) {
        printf("epoch %d\n", (i + 1));
        accuracyListPrint(MODULAR_COUNT + 1, trainAccuracyList[i]);
    }

    // 시험 결과 출력
    double **testAccuracyList = test(mlp, (int) (NUMBER_LIMIT * (1 - TRAIN_RATE)), testInputDataList,
                                     testAnswerIndexList);
    printf("Test\n");
    accuracyListPrint(MODULAR_COUNT + 1, testAccuracyList);

    // free
    free(data);
    for (i = 0; i < NUMBER_LIMIT; ++i)
        free(inputDataList[i]);
    free(inputDataList);
    free(answerIndexList);
    for (i = 0; i < BINARY_DIGIT_LENGTH; ++i)
        free(trainInputDataList[i]);
    free(trainInputDataList);
    free(trainAnswerIndexList);
    for (i = 0; i < BINARY_DIGIT_LENGTH; ++i)
        free(testInputDataList[i]);
    free(testInputDataList);
    free(testAnswerIndexList);
    for (i = 0; i < TRAIN_COUNT; ++i) {
        for (j = 0; j < MODULAR_COUNT + 1; ++j)
            free(trainAccuracyList[i][j]);
        free(trainAccuracyList[i]);
    }
    free(trainAccuracyList);
    for (i = 0; i < MODULAR_COUNT; ++i)
        free(testAccuracyList[i]);
    free(testAccuracyList);
    deleteMLP(mlp);
    return 0;
}

double **getInputDataList(int dataLength, int *data) {
    double **inputDataList = (double **) malloc(sizeof(double *) * dataLength);
    int dataIndex, binaryIndex, nowValue, binaryValue;
    for (dataIndex = 0; dataIndex < dataLength; ++dataIndex) {
        inputDataList[dataIndex] = (double *) malloc(sizeof(double) * BINARY_DIGIT_LENGTH);
        nowValue = data[dataIndex];
        for (binaryIndex = BINARY_DIGIT_LENGTH - 1; binaryIndex >= 0; --binaryIndex) {
            binaryValue = pow(2, binaryIndex);
            if (nowValue >= binaryValue) {
                nowValue -= binaryValue;
                inputDataList[dataIndex][binaryIndex] = 1;
            } else
                inputDataList[dataIndex][binaryIndex] = 0;
        }
    }
    return inputDataList;
}

int *getAnswerIndexList(int dataLength, int *data) {
    int dataIndex, *answerIndexList = (int *) malloc(sizeof(int) * dataLength);
    for (dataIndex = 0; dataIndex < dataLength; ++dataIndex)
        answerIndexList[dataIndex] = data[dataIndex] % NEURON_COUNT[LAYER_COUNT - 1];
    return answerIndexList;
}

void accuracyListPrint(int listLength, double **accuracyList) {
    int index;
    for (index = 0; index < listLength - 1; ++index)
        printf("\tAccuracy %d = %d/%d = %lf\n", index, (int) accuracyList[index][0], (int) accuracyList[index][1],
               accuracyList[index][2]);
    printf("\tTotal accuracy = %d/%d = %lf\n", (int) accuracyList[listLength - 1][0],
           (int) accuracyList[listLength - 1][1], accuracyList[listLength - 1][2]);
}
