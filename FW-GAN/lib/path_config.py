ImgHeight = 32

data_roots = {
    'iam_word': './data/',
    # Repo default: a single "custom" dataset built from ./dataset -> ./data/*.hdf5
    # using build_hdf5.py and the Spanish-compatible alphabet in lib/alphabet.py.
    'custom': './data/'
}

data_paths = {
    'iam_word': {'trnval': 'train.hdf5',
                 'test': 'test.hdf5'},
    'iam_word_org': {'trnval': 'train.hdf5',
                     'test': 'test.hdf5'}
    ,
    # Alias so configs can set dataset: 'custom' without breaking get_dataset().
    'custom': {'trnval': 'train.hdf5',
               'test': 'test.hdf5'},
    'custom_org': {'trnval': 'train.hdf5',
                   'test': 'test.hdf5'}
}

split_files = {
    'custom': {
        'train': './splits/train.txt',
        'val': './splits/val.txt',
        'test': './splits/test.txt',
    }
}
