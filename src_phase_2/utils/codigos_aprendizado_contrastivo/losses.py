import torch
import torch.nn.functional as F


# Contastive Loss with Cosine Similarity

class ContrastiveLossCos(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=2.0, contrastive_thres=1.1):
        super(ContrastiveLossCos, self).__init__()
        self.contrastive_thres=contrastive_thres
        self.margin = margin
    def forward(self, output1, output2, label):
        cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        cos_sim = cos(output1, output2)
        loss_contrastive = torch.mean((1-label) * torch.pow(cos_sim, 2)/2 +
            (label) * torch.pow(torch.clamp(self.margin - cos_sim, min=0.0), 2))/2
        acc = ((cos_sim > self.contrastive_thres) == label).float().mean()
        return loss_contrastive, acc

# Contastive Loss

class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=2.0, contrastive_thres=1.1):
        super(ContrastiveLoss, self).__init__()
        self.contrastive_thres=contrastive_thres
        self.margin = margin
    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim = True)
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2)/2 +
            (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))/2
        acc = ((euclidean_distance > self.contrastive_thres) == label).float().mean()
        return loss_contrastive, acc

# Contastive Loss Chopra

class ContrastiveLossChopra(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/chopra-05.pdf
    """

    def __init__(self, Q=2.0, contrastive_thres=1.1):
        super(ContrastiveLossChopra, self).__init__()
        self.contrastive_thres=contrastive_thres
        self.Q = Q
    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim = True)
        loss_contrastive = torch.mean((1-label) * (2 / self.Q) * torch.pow(euclidean_distance, 2)/2 +
            (label) * (2 * self.Q) * torch.exp((-2.77 / self.Q) * euclidean_distance)) / 2
        acc = ((euclidean_distance > self.contrastive_thres) == label).float().mean()
        return loss_contrastive, acc

# Triplet Loss

class TripletLoss(torch.nn.Module):
    """
    Triplet loss function.
    """

    def __init__(self, margin=2.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):

        distance_1 = F.pairwise_distance(anchor, positive, keepdim = True)
        distance_2 = F.pairwise_distance(anchor, negative, keepdim = True)
        
        triplet_loss = F.relu(torch.pow(distance_1, 2) - torch.pow(distance_2, 2) + self.margin).mean()
        acc = (distance_1 < distance_2).sum() * 1.0 / distance_1.size()[0]
        
        return triplet_loss, acc

# Quadruplet Loss

class QuadrupletLoss(torch.nn.Module):
    """
    Quadruplet loss function.
    """

    def __init__(self, margin1=2.0, margin2=1.0):
        super(QuadrupletLoss, self).__init__()
        self.margin1 = margin1
        self.margin2 = margin2

    def forward(self, anchor, positive, negative1, negative2):

        distance_1 = F.pairwise_distance(anchor, positive, keepdim = True)
        distance_2 = F.pairwise_distance(anchor, negative1, keepdim = True)
        distance_3 = F.pairwise_distance(negative1, negative2, keepdim = True)
        
        quadruplet_loss = F.relu(torch.pow(distance_1, 2) - torch.pow(distance_2, 2) + self.margin1).mean() \
                            + F.relu(torch.pow(distance_1, 2) - torch.pow(distance_3, 2) + self.margin2).mean()
        acc = ((distance_1 < distance_2) * (distance_1 < distance_3)).sum() * 1.0 / distance_1.size()[0]
        
        return quadruplet_loss, acc