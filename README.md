# aics-project title: A Siamese Network for Learning Multimodal Similarity in the WikiDiverse Dataset
## An overview
Siamese networks consist of two identical sub-networks that share weights and learn to compute the similarity between two input samples. The goal is to learn embeddings such that similar inputs are close in the embedding space, while dissimilar inputs are far apart. For the WikiDiverse dataset, where we have image-caption pairs, we can build a Siamese network that processes text and image data (or just one modality like text or image) and learns to compute similarity between two entities from the knowledge base.
## * Siamese Network Structure: Two identical sub-networks that compute embeddings for input pairs and learn their similarity
## * Application: For WikiDiverse, compute similarity between image-caption pairs to link knowledge-base entities.

## Contents of this Repository
-[Scripts](#Scripts) 
 Python scripts for training, evaluation, and analysis.
-[Notes](#Notes) 
 Figure, training results(metrics and verification), and other.
-[Paper](#Paper)
 The full paper detailing methodology, results, and discussions.
## Since the training dataset and models are large in size, I have uploaded them to the following Google Drive link:
-[Link](#Link) https://drive.google.com/drive/folders/1BCFgt429xTY2nM1ZHMTGFPJmfgvRVYyz
