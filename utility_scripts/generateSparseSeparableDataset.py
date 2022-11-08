from random import random

NUM_DATAPOINTS: int = 2048
NUM_FEATURES: int = 2048

TARGET_DENSITY: float = 0.1
ZERO_THRESHOLD_FRACTION: float = 0#0.99
SPLIT: float = 0.9

DATA_MAX_VALUE: float = 100.
DATA_MIN_VALUE: float = -100.

FILEPATH: str = "E:\SparseDatasets\\density\\8192_8192_1_A.libsvm"


def write_data_point(classifier: int, data: list[float]):
    with open(FILEPATH, 'a+') as libsvmFile:
        line: str = str(classifier) + " "
        for i in range(0, len(data)):
            if not data[i] == 0.:
                line += str(i + 1) + ":" + str(data[i]) + " "
        libsvmFile.write(line + "\n")


if __name__ == '__main__':
    nnz: int = 0

    # generate random thresholds for each dimension (to ensure linear separability)
    thresholds: list[float] = [0.] * NUM_FEATURES
    for i in range(NUM_FEATURES):
        if random() > ZERO_THRESHOLD_FRACTION:
            thresholds[i] = DATA_MIN_VALUE + random() * (DATA_MAX_VALUE - DATA_MIN_VALUE)

    # classify datapoints (as either 1 or -1)
    classifiers: list[int] = [1] * NUM_DATAPOINTS
    for i in range(NUM_DATAPOINTS):
        classifiers[i] = 1 if random() > SPLIT else -1
    onesFraction = classifiers.count(1) / NUM_FEATURES

    # generate random feature values for each datapoint
    for i in range(NUM_DATAPOINTS):
        datapoint: list[float] = [0.0] * NUM_FEATURES
        for j in range(NUM_FEATURES):
            if classifiers[i] == 1:
                if random() > TARGET_DENSITY * onesFraction * 2 and 0. >= thresholds[j]:
                    pass
                else:
                    datapoint[j] = thresholds[j] + random() * (DATA_MAX_VALUE - thresholds[j])
                    nnz += 1
            else:
                if random() > TARGET_DENSITY * (1 - onesFraction) * 2 and 0. <= thresholds[j]:
                    pass
                else:
                    datapoint[j] = DATA_MIN_VALUE + random() * (thresholds[j] - DATA_MIN_VALUE)
                    nnz += 1
        write_data_point(classifiers[i], datapoint)

    # add comment with metadata at end of file
    with open(FILEPATH, 'a+') as libsvmFile:
        libsvmFile.write("# Data Points: " + str(NUM_DATAPOINTS) + "\n")
        libsvmFile.write("# Features: " + str(NUM_FEATURES) + "\n")
        libsvmFile.write("# Total Entries: " + str(NUM_DATAPOINTS * NUM_FEATURES) + "\n")
        libsvmFile.write("# Density: " + str(nnz / (NUM_DATAPOINTS * NUM_FEATURES)) + "\n")
        libsvmFile.write("# Separability: linear" + "\n")
    print("# Data Points: " + str(NUM_DATAPOINTS))
    print("# Features: " + str(NUM_FEATURES))
    print("# Total Entries: " + str(NUM_DATAPOINTS * NUM_FEATURES))
    print("# Density: " + str(nnz / (NUM_DATAPOINTS * NUM_FEATURES)))
    print("# Separability: linear")
