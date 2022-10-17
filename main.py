from utils import createDataSplitSet, plot_train_data, compile_train_get_results
from model import feature_extractor, fine_tune
import argparse

#globals
IMG_SIZE = 224
LR = 1e-4



train_dir = 'data/covid19_CT/train'
validation_dir = 'data/covid19_CT/validation'
test_dir = 'data/covid19_CT/test'



#creating training, testing, and validation data
train_X, train_y = createDataSplitSet(train_dir)
val_X, val_y = createDataSplitSet(validation_dir)
test_X, test_y = createDataSplitSet(test_dir)



if __name__ == '__main__':
    arg = argparse.ArgumentParser()
    arg.add_argument("-pd", "--plotData", required=False, help="add a number from 0 to 2500")
    arg.add_argument("-t", "--typeOfClassifier", required=True, help="Select fe for feature extraction. Select ft for fine-tune.")
    args = vars(arg.parse_args())

    if args['plotData']:
        plot_train_data(args['plotData'])
    elif args['typeOfClassifier'] == 'fe':
        model = feature_extractor(train_X, train_y, val_X, val_y)
        compile_train_get_results(model, train_X, train_y, val_X, val_y, test_X, test_y)
    elif args['typeOfClassifier'] == 'ft':
        model = fine_tune(train_X, train_y, val_X, val_y)
        compile_train_get_results(model, train_X, train_y, val_X, val_y, test_X, test_y)
    else:
        print("Incorrect Module: Type --help with file name to see options.")