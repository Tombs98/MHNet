import os
from dataset_RGB import DataLoaderTrain, DataLoaderVal, DataLoaderTest, DataLoaderTest2

def get_training_data(rgb_dir, img_options):
    assert os.path.exists(rgb_dir)
    return DataLoaderTrain(rgb_dir, img_options)

def get_validation_data(rgb_dir, img_options):
    assert os.path.exists(rgb_dir)
    return DataLoaderVal(rgb_dir, img_options)

def get_test_data(rgb_dir, img_options):
    assert os.path.exists(rgb_dir)
    return DataLoaderTest(rgb_dir, img_options)

def get_test_data2(rgb_dir, img_options):
    assert os.path.exists(rgb_dir)
    return DataLoaderTest2(rgb_dir, img_options)


def debu():
    from config import Config
    opt = Config('training.yml')
    train_dir = opt.TRAINING.TRAIN_DIR
    train_dataset = get_training_data(train_dir, {'patch_size': opt.TRAINING.TRAIN_PS})
    test  = os.path.join('./Datasets/', 'GoPr', 'test', 'input')
    test_dataset = get_test_data(test, {'patch_size': opt.TRAINING.VAL_PS})
    print("1", len(test_dataset))
    print(len(test_dataset[0]))
    print(test_dataset[0][0].size())
# debu()
