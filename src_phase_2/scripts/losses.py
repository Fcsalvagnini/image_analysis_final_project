from torch_snippets import F
import torch


class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=2.0, contrastive_threshold=1.1):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.contrastive_threshold = contrastive_threshold

    def forward(self, output_1, output_2, label):
        euclidean_distance = F.pairwise_distance(
            output_1, output_2, keepdim=True
        )
        loss_contrastive = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) / 2 +
                                      label * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0),
                                                        2)) / 2
        acc = ((euclidean_distance > self.contrastive_threshold) == label).float().mean()
        return loss_contrastive, acc


class ContrastiveCosineLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=2.0, contrastive_threshold=1.1):
        super(ContrastiveCosineLoss, self).__init__()
        self.margin = margin
        self.contrastive_threshold = contrastive_threshold

    def forward(self, output_1, output_2, label):
        euclidean_distance = 1 - F.cosine_similarity(
            output_1, output_2
        )
        loss_contrastive = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) / 2 +
                                      label * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0),
                                                        2)) / 2
        acc = ((euclidean_distance > self.contrastive_threshold) == label).float().mean()
        return loss_contrastive, acc


class TripletLoss(torch.nn.Module):
    def __init__(self, margin=2.0):
        super(TripletLoss, self).__init__()
        self.triplet_margin_loss = torch.nn.TripletMarginLoss(margin=margin)

    def forward(self, output_anchor, output_pos, output_neg):
        loss = self.triplet_margin_loss(output_anchor, output_pos, output_neg)
        # -----adjust output---------------------------------
        return loss, loss  # << adjust loss


class CosineLoss(torch.nn.Module):
    def __init__(self, margin=2.0, contrastive_threshold=1.1):
        super(CosineLoss, self).__init__()
        self.CosineLoss = torch.nn.CosineEmbeddingLoss(margin=margin)
        self.contrastive_threshold = contrastive_threshold

    def forward(self, output_1, output_2, labels):
        loss = self.CosineLoss(output_1, output_2, labels)
        # -----adjust output---------------------------------
        return loss, loss  # << adjust loss
