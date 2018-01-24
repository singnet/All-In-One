from util.preprocess import ImdbWikiDatasetPreprocessor
from util.nets import AllInOneNeuralNetwork
import argparse

INPUT_SIZE = (227,227,3)


# preprocessor = ImdbWikiDatasetPreprocessor("/home/mtk/datasets/wiki","wiki")
# preprocessor = ImdbWikiDatasetPreprocessor("/home/mtk/datasets/israel/imdb_crop","imdb")
# preprocessor.split_dataset_to_train_test(0.8)
# preprocessor.train_model(None)
def main():
    parser = argparse.ArgumentParser()
    # --mtype is model type argument. it can be either 'np'(neutral vs positive emotion classifier) or 'ava'(All basic
    # seven emotion classifier[anger,fear,disgust,happy,sad,surprise,neutral]). Default is 'np'
    parser.add_argument("--images_path")
    parser.add_argument("--dataset")
    args = parser.parse_args()

    images_path = args.images_path
    dataset = args.dataset
    preprocessor = ImdbWikiDatasetPreprocessor(images_path,dataset)
    net= AllInOneNeuralNetwork(INPUT_SIZE,preprocessor)
    net.train()

if __name__== "__main__":
    main()