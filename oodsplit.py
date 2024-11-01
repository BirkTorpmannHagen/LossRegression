# import matplotlib.pyplot as plt
# import numpy as np
# import torch
# from domain_datasets import build_nico_dataset
# from torch.utils.data import DataLoader
# from vae.models.vanilla_vae import VanillaVAE
# from vae.vae_experiment import VAEXperiment
# import yaml
# import torchvision.transforms as transforms
# from sklearn.mixture import  BayesianGaussianMixture
# from sklearn.cluster import KMeans
# from yellowbrick.features import Manifold
# from tqdm import tqdm
# from classifier.resnetclassifier import ResNetClassifier
# import os
# from sklearn.decomposition import PCA
# from sklearn.manifold import TSNE
# from sklearn.cluster import KMeans
# from torch.utils.data import Sampler
# class IndexSampler(Sampler):
#     def __init__(self, data_source, indices):
#         super(IndexSampler, self).__init__(data_source)
#         self.indices = indices
#
#     def __iter__(self):
#         return iter(self.indices)
#
#     def __len__(self):
#         return len(self.indices)
#
# class VAEOODSplitter():
#     def __init__(self, dataset, split=(80, 10, 10), vae_dim=8000):
#         assert len(split)==3, "split must be threefold, [%training data, %validaation data, %test data]"
#         self.dataset = dataset
#         latents = torch.zeros([5000, vae_dim])
#         vae_model = VanillaVAE(3, 8000).to("cuda")
#         config = yaml.safe_load("vae/configs/vae.yaml")
#         vae_exp = VAEXperiment(vae_model, config)
#         vae_exp.load_state_dict(torch.load("logs/VanillaVAE/version_14/checkpoints/last.ckpt")["state_dict"])
#         cluster = KMeans(n_clusters=8)
#         contexts = []
#         labels = []
#         dataloader = DataLoader(dataset, shuffle=False)
#         for i, (x, y, context) in tqdm(enumerate(dataloader), total=len(dataloader)):
#             with torch.no_grad():
#                 latents[i]=vae_model.encode(x.to("cuda"))[0]
#             contexts.append(context.cpu().numpy()[0])
#             labels.append(y.cpu().numpy()[0])
#         clustering = KMeans(10)
#         self.folds = clustering.fit_predict(latents)
# #        visualizer.fit_transform(latents,contexts)
#  #       visualizer.show()
#
#     def get_trainloader(self):
#         indices = np.arange(len(self.dataset))[self.folds > 1]
#         np.random.shuffle(indices)
#         return DataLoader(self.dataset, sampler=IndexSampler(data_source=self.dataset, indices=indices))
#
#     def get_valloader(self):
#         indices = np.arange(len(self.dataset))[self.folds == 1]
#         np.random.shuffle(indices)
#         return DataLoader(self.dataset, sampler=IndexSampler(data_source=self.dataset, indices=indices))
#
#     def get_testloader(self):
#         indices = np.arange(len(self.dataset))[self.folds == 0]
#         np.random.shuffle(indices)
#         return DataLoader(self.dataset, sampler=IndexSampler(data_source=self.dataset, indices=indices))
#
#
# class FeatureOODSplitter:
#     def __init__(self, dataset, split=(80, 10, 10), vae_dim=8000):
#         self.dataset = dataset
#         assert len(split) == 3, "split must be threefold, [%training data, %validaation data, %test data]"
#         nc = len((os.listdir("datasets/NICO++/track_1/public_dg_0416/train/autumn")))
#         model = ResNetClassifier.load_from_checkpoint("lightning_logs/version_0/checkpoints/epoch=19-step=1599600.ckpt",num_classes = nc, resnet_version=34, gpus=[0])
#         newmodel = torch.nn.Sequential(*(list(model.resnet_model.children())[:-1])).cuda()
#         self.features = np.zeros((len(dataset), 512))
#         contexts = np.zeros((len(dataset), 1))
#         classes = np.zeros((len(dataset), 1))
#         dataloader = DataLoader(dataset, shuffle=False)
#
#         for i, (x, y, context) in tqdm(enumerate(dataloader), total=len(dataloader)):
#             x = x.cuda()
#             with torch.no_grad():
#                 self.features[i] = newmodel(x).squeeze(-1).squeeze(-1).cpu().numpy()
#                 contexts[i]=int(context)
#                 classes[i]=int(y)
#         clustering = KMeans(10)
#         self.folds = clustering.fit_predict(self.features)
#         pca = TSNE(2)
#         transformed = pca.fit_transform(self.features)
#         plt.scatter(transformed[:, 0], transformed[:,1], c=contexts, cmap="viridis")
#         plt.show()
#         plt.scatter(transformed[:, 0], transformed[:, 1], c=classes, cmap="viridis")
#         plt.show()
#
#     def get_trainloader(self):
#         indices = np.arange(len(self.dataset))[self.folds>1]
#         np.random.shuffle(indices)
#         return DataLoader(self.dataset, sampler=IndexSampler(data_source=self.dataset, indices=indices))
#
#     def get_valloader(self):
#         indices = np.arange(len(self.dataset))[self.folds==1]
#         np.random.shuffle(indices)
#         return DataLoader(self.dataset, sampler=IndexSampler(data_source=self.dataset, indices=indices))
#
#     def get_testloader(self):
#         indices = np.arange(len(self.dataset))[self.folds == 0]
#         np.random.shuffle(indices)
#         return DataLoader(self.dataset, sampler=IndexSampler(data_source=self.dataset, indices=indices))
#
# if __name__ == '__main__':
#     trans = transforms.Compose([transforms.RandomHorizontalFlip(),
#                         transforms.Resize((512,512)),
#                         transforms.ToTensor(), ])
#     train_set = build_nico_dataset(1, "datasets/NICO++", 0, trans, trans, 0)[0]
#     # splitter = VAEOODSplitter(dloader)
#     splitter = FeatureOODSplitter(train_set)
#     for x,y,context in splitter.get_trainloader():
#         print(context)