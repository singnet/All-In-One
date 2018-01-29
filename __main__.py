from util.preprocess import ImdbWikiDatasetPreprocessor,CelebADatasetPreprocessor
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
    parser.add_argument("--lr",default=1e-4,type=float)
    parser.add_argument("--load_db",default=False,type=bool)
    parser.add_argument("--resume",default = False,type=bool)
    parser.add_argument("--steps",default = 100,type=int)
    parser.add_argument("--ol",default = "large_model",type=str)
    parser.add_argument("--os",default = "",type=str)

    args = parser.parse_args()
    if not os.path.exists(args.images_path):
        print "image path given does not exists"
        exit(0)
    if not args.dataset in ["wiki","imdb", "celeba"]:
        print "currently implemented for only wiki, imdb and celeba datasets"
        exit(0)
    images_path = args.images_path
    dataset = args.dataset

    if dataset=="wiki" or dataset == "imdb":
        if args.os=="":
            small_model = "wiki_age_gender"
        else:
            small_model = args.os
        preprocessor = ImdbWikiDatasetPreprocessor(images_path,dataset,args.load_db)
    elif dataset=="celeba":
        preprocessor = CelebADatasetPreprocessor(images_path)
        if args.os=="":
            small_model = "celeba_smile"
        else:
            small_model = args.os
        
    else:
        print "Unable to recognize the dataset type given",dataset
        exit(0)
    net= AllInOneNeuralNetwork(INPUT_SIZE,preprocessor,batch_size=args.batch_size,epochs=args.epochs,learning_rate=args.lr,load_db = args.load_db,resume=args.resume,steps_per_epoch=args.steps,large_model_name=args.ol,small_model_name=small_model)
    net.train()

if __name__== "__main__":
    main()