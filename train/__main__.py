from nets import AllInOneNetwork
from dataset.celeba import CelebAAlignedDataset
from dataset.imdb_wiki import ImdbWikiDataset
from util import get_cmd_args
from util import get_config
import os

INPUT_SIZE = (227,227,1)

def main():
    args = get_cmd_args()
    if not os.path.exists(args.images_path):
        print "image path given does not exists"
        exit(0)
    if not args.dataset.lower() in ["wiki","imdb", "celeba","aflw", "adience"]:
        print "currently implemented for only wiki, imdb, aflw and celeba datasets"
        exit(0)
    config = get_config(args)
    net = AllInOneNetwork(config)
    net.train()


if __name__=="__main__":
    main()
