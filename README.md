Multirelational Twitter Bot Detection using Graph Neural Networks

NOTE: This code has been developed using the code from https://github.com/LuoUndergradXJTU/TwiBot-22/tree/master/src/Wei as base

Steps to reproduce the code and results.
- Request access to the Twibot-22 dataset from the original author.
- To access the large dataset without downloading it, create a shortcut on your drive and access the shortcut path on your drive on colab.
- Since the dataset is large, we might have to process it on virtual machines on cloud.
- Load the dataset on Google Cloud Storage (in case of GCP, use equivalent services on other cloud platforms).
- Use dataproc to filter/split large files depending on your requirements.

Combinations of heterogenous relations in the edges.csv have also been used to construct homogenous relations to be processed.

We have the dataset (user.json, edge.csv, label.csv, split.csv)
- Run dataset-preprocessing.ipynb to pre-process the dataset for filtering, balancing the user, label and split.
- Run edge-preprocessing.ipynb to pre-process the edges. Make changes to the code to include more relations, or combinations for your purpose.
- Run feature-preprocessing.ipynb to extract feature vectors to be used as embeddings for GNN algorithms. Extracted numerical properties, categorical properties, user description from the dataset.
- GNN models are included in the model files. (Four models; GCN, RGCN, GAT and GraphSAGE are used).
- Train the model on different combinations of relations using the training files.
