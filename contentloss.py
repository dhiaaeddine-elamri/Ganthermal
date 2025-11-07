import torch
import torch.nn as nn

class ContentLoss(nn.Module):
    def __init__(self):
        super(ContentLoss, self).__init__()
        self.l1 = nn.L1Loss(reduction='mean')  # automatically does the 1/(W*H)

    def forward(self, generated_image, ground_truth_image):

        return self.l1(generated_image, ground_truth_image)
    