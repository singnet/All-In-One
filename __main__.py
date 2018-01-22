from util.nets import AllInOneNeuralNetwork
from util.preprocess import getAgeGenderDataset
from util.preprocess_celba import load_dataset


INPUT_SIZE = (227,227,3)

net = AllInOneNeuralNetwork(INPUT_SIZE)
# model = net.model
# print net.get_face_reco_model().summary()
# for i in range(len(model.layers)-8,len(model.layers)):
#     print model.layers[i].name
# print len(model.layers)

# load_wiki_dataset("/home/mtk/datasets/wiki/",["score"])
# test_age_of_wiki_dataset("/home/mtk/datasets/wiki")
# test_selective_search("imgs/44.jpg")
# getAgeGenderDataset("/home/mtk/datasets/wiki","/home/mtk/datasets/israel/imdb_crop")

# net.train()

load_dataset("","/home/mtk/datasets/csia/list_attr_celeba.txt")