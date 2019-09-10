import os
import random
import configparser

def main(data_config_path):
    data_directory = []
    classes_names_list = []
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

    # ** load data.ini
    data_config = configparser.ConfigParser()
    data_config.read(data_config_path)

    # ** get all data proportion
    train_prop = data_config.getfloat('training', 'proportion')
    val_prop = data_config.getfloat('validation', 'proportion')
    test_prop = data_config.getfloat('testing', 'proportion')

    # NOTE split all data by each proportion
    original_data_path = data_config.get('data', 'path')

    # -- classes names save at classes_names.txt --
    classes_names_list = os.listdir(original_data_path)
    for folder_name in os.listdir(original_data_path):
        for image_name in os.listdir(os.path.join(original_data_path, folder_name)):
            if '.jpg' in image_name:
                class_num = classes_names_list.index(folder_name)
                data_directory.append([os.path.join(original_data_path, folder_name, image_name), str(class_num)])

    # NOTE shuffle
    random.shuffle(data_directory)

    # NOTE split dataset
    n_sample = len(data_directory)

    train_idx = int(train_prop * n_sample)
    val_idx = int(val_prop * n_sample) + train_idx
    test_idx  = int(test_prop  * n_sample) + val_idx

    train_indices = data_directory[:train_idx]
    val_indices = data_directory[train_idx:val_idx]
    test_indices = data_directory[val_idx:]

    # ** training
    with open(data_config.get('training', 'filename'), 'w') as f:
        for [img_path, ann_num] in train_indices:
            f.write(img_path)
            f.write(' ')
            f.write(ann_num)
            f.write('\n')

    # ** validation
    with open(data_config.get('validation', 'filename'), 'w') as f:
        for [img_path, ann_num] in val_indices:
            f.write(img_path)
            f.write(' ')
            f.write(ann_num)
            f.write('\n')

    # ** testing
    with open(data_config.get('testing', 'filename'), 'w') as f:
        for [img_path, ann_num] in test_indices:
            f.write(img_path)
            f.write(' ')
            f.write(ann_num)
            f.write('\n')

    # NOTE save classes names
    with open(data_config.get('data', 'classes_path'), 'w') as f:
        for class_name in classes_names_list:
            f.write(class_name)
            f.write('\n')
    
    # NOTE print finish message
    print('class names(labels):', classes_names_list)
    print('train data:', len(train_indices))
    print('val data:', len(val_indices))
    print('test data:', len(test_indices))
    print('all data:', n_sample)
    print('data is prepared~')

if __name__ == '__main__':
    data_config_path = 'config/data.ini'
    main(data_config_path)