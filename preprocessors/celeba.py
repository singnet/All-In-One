from preprocessors import AbastractPreprocessor


class CelebAPreprocessor(AbastractPreprocessor):
    """Class that preprocesses CelebA dataset.
    for more information visit preprocessors.AbastractPreprocessor
    """

    def __init__(self,dataset_type,dataset_dir,split=False):
        super(CelebAPreprocessor,self).__init__(dataset_type,dataset_dir,split=split)
    
    def fix_labeling_issue(self,dataset):
        pass
    def generator(self,batch_size=32):
        pass