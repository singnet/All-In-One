# from util.preprocess import ImdbWikiDatasetPreprocessor,CelebADatasetPreprocessor
# from util.nets import AllInOneNetwork
from nets import AllInOneNetwork
from dataset.celeba import CelebAAlignedDataset
import argparse
import os

INPUT_SIZE = (227,227,3)


# preprocessor = ImdbWikiDatasetPreprocessor("/home/mtk/datasets/wiki","wiki")
# preprocessor = ImdbWikiDatasetPreprocessor("/home/mtk/datasets/israel/imdb_crop","imdb")
# preprocessor.split_dataset_to_train_test(0.8)
# preprocessor.train_model(None)
def get_dataset(name,dataset_dir):
    if name=="celeba":
        return CelebAAlignedDataset(dataset_dir=dataset_dir,labels=["Smiling"])
    
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
    parser.add_argument("--load_model",default=None,type=str)

    args = parser.parse_args()
    if not os.path.exists(args.images_path):
        print "image path given does not exists"
        exit(0)
    if not args.dataset in ["wiki","imdb", "celeba"]:
        print "currently implemented for only wiki, imdb and celeba datasets"
        exit(0)

    
    datasetClass = get_dataset(args.dataset,args.images_path)
    datasetClass.load_dataset()
    net = AllInOneNetwork((227,227,3),datasetClass,epochs=args.epochs,batch_size= args.batch_size,learning_rate=args.lr,load_db=args.load_db,resume=args.resume,
        steps_per_epoch=args.steps,large_model_name=args.ol,small_model_name=args.os,load_model=args.load_model)
    net.train()

if __name__== "__main__":
    main()