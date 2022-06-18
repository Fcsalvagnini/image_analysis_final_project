To execute the SiNN with Contrastive Loss for face verification

$ python3 siamese_contrastive_faces.py

To execute the SiNN with Triplet Loss for face verification

$ python3 siamese_triplet_faces.py

To execute the SiNN with Quadruplet Loss for face verification

$ python3 siamese_quadruplet_faces.py

=======================================================================

To execute the SiNN with Contrastive Loss for corel

$ python3 siamese_contrastive_corel.py

To execute the SiNN with Triplet Loss for corel

$ python3 siamese_triplet_corel.py

To execute the SiNN with Quadruplet Loss for corel

$ python3 siamese_quadruplet_corel.py

These calls will generate the .pth models. Once these files have been generated it is possible to run the corel_transfer.py

======================================================================

To execute the MLP classifier with backbone SiNN with Contrastive Loss

$ python3 corel_transfer.py model_contrastive_corel.pth

To execute the MLP classifier with backbone SiNN with Triplet Loss

$ python3 corel_transfer.py model_triplet_corel.pth

To execute the MLP classifier with backbone SiNN with Quadruple Loss

$ python3 corel_transfer.py model_quadruple_corel.pth
