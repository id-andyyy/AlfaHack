import pandas as pd
import glob
from sklearn.decomposition import PCA

def load_and_preprocess_data(path_train='train', path_test='test'):
    filenames_train = glob.glob(path_train + "/*.csv")
    
    pca = PCA(n_components=50, random_state=42)
    learn = pd.read_csv(filenames_train[0])
    pca.fit(learn.drop(['target', 'smpl', 'id'], axis=1))
    new_columns = [f'feature_{i}' for i in range(1, 51)]
    
    def compression(filename, features=['target', 'smpl', 'id']):
        data = pd.read_csv(filename)
        base_info = data[features]
        transformed_data = pd.DataFrame(pca.transform(data.drop(features, axis=1)))
        result = pd.concat([base_info, transformed_data], ignore_index=True, axis=1)
        result.columns = [*features, *new_columns]
        return result
    
    data_files_train = []
    for filename in filenames_train:
        data_files_train.append(compression(filename))
    train_data = pd.concat(data_files_train, ignore_index=True)
    
    filenames_test = glob.glob(path_test + "/*.csv")
    data_files_test = []
    for filename in filenames_test:
        data_files_test.append(compression(filename, features=['smpl', 'id']))
    test_data = pd.concat(data_files_test, ignore_index=True)
    
    return train_data, test_data