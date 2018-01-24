from util.preprocess import ImdbWikiDatasetPreprocessor
from util.nets import AllInOneNeuralNetwork

INPUT_SIZE = (227,227,3)


# preprocessor = ImdbWikiDatasetPreprocessor("/home/mtk/datasets/wiki","wiki")
# preprocessor = ImdbWikiDatasetPreprocessor("/home/mtk/datasets/israel/imdb_crop","imdb")
# preprocessor.split_dataset_to_train_test(0.8)
# preprocessor.train_model(None)

net= AllInOneNeuralNetwork(INPUT_SIZE)
net.train()
