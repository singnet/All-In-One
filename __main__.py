from util.preprocess import ImdbWikiDatasetPreprocessor
from util.nets import AllInOneNeuralNetwork
import argparse
import os

INPUT_SIZE = (227,227,3)


# preprocessor = ImdbWikiDatasetPreprocessor("/home/mtk/datasets/wiki","wiki")
# preprocessor = ImdbWikiDatasetPreprocessor("/home/mtk/datasets/israel/imdb_crop","imdb")
# preprocessor.split_dataset_to_train_test(0.8)
# preprocessor.train_model(None)
def main():
    parser = argparse.ArgumentParser()
    # --mtype is model type argument. it can be either 'np'(neutral vs positive emotion classifier) or 'ava'(All basic
    # seven emotion classifier[anger,fear,disgust,happy,sad,surprise,neutral]). Default is 'np'
    parser.add_argument("--images_path",default="",type=str)
    parser.add_argument("--dataset",default="",type=str)
    parser.add_argument("--epochs",default=10,type=int)
    parser.add_argument("--batch_size",default=32,type=int)
    args = parser.parse_args()
    if not os.path.exists(args.images_path):
        print "image path given does not exists"
        exit(0)
    if not args.dataset in ["wiki","imdb"]:
        print "currently implemented for only wiki and imdb datasets"
        exit(0)
    images_path = args.images_path
    dataset = args.dataset
    preprocessor = ImdbWikiDatasetPreprocessor(images_path,dataset)
    net= AllInOneNeuralNetwork(INPUT_SIZE,preprocessor,batch_size=args.batch_size,epochs=args.epochs)
    net.train()

if __name__== "__main__":
    main()