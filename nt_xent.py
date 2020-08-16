import torch
import torch.nn.functional as F

import torch
import numpy as np


class NT_Xent(torch.nn.Module):

    def __init__(self, device, batch_size, temperature, use_cosine_similarity):
        super(NT_Xent, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device
        self.softmax = torch.nn.Softmax(dim=-1)
        self.mask_samples_from_same_repr = self._get_correlated_mask().type(torch.bool)
        self.similarity_function = self._get_similarity_function(use_cosine_similarity)
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    def _get_similarity_function(self, use_cosine_similarity):
        if use_cosine_similarity:
            self._cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
            return self._cosine_simililarity
        else:
            return self._dot_simililarity

    def _get_correlated_mask(self):
        diag = np.eye(2 * self.batch_size)
        l1 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=-self.batch_size)
        l2 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=self.batch_size)
        mask = torch.from_numpy((diag + l1 + l2))
        mask = (1 - mask).type(torch.bool)
        return mask.to(self.device)

    @staticmethod
    def _dot_simililarity(x, y):
        v = torch.tensordot(x.unsqueeze(1), y.T.unsqueeze(0), dims=2)
        # x shape: (N, 1, C)
        # y shape: (1, C, 2N)
        # v shape: (N, 2N)
        return v

    def _cosine_simililarity(self, x, y):
        # x shape: (N, 1, C)
        # y shape: (1, 2N, C)
        # v shape: (N, 2N)
        v = self._cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        return v

    def forward(self, zis, zjs):
        representations = torch.cat([zjs, zis], dim=0)

        similarity_matrix = self.similarity_function(representations, representations)

        # filter out the scores from the positive samples
        l_pos = torch.diag(similarity_matrix, self.batch_size)
        r_pos = torch.diag(similarity_matrix, -self.batch_size)
        positives = torch.cat([l_pos, r_pos]).view(2 * self.batch_size, 1)

        negatives = similarity_matrix[self.mask_samples_from_same_repr].view(2 * self.batch_size, -1)

        logits = torch.cat((positives, negatives), dim=1)
        logits /= self.temperature

        labels = torch.zeros(2 * self.batch_size).to(self.device).long()
        loss = self.criterion(logits, labels)

        return loss / (2 * self.batch_size)

'''

def cosine_pairwise(x):
    x = x.permute((1, 2, 0))
    cos_sim_pairwise = F.cosine_similarity(x, x.unsqueeze(1), dim=-2)
    cos_sim_pairwise = cos_sim_pairwise.permute((2, 0, 1))
    return cos_sim_pairwise


class NT_Xent(torch.nn.Module):

	def __init__(self, batch_size, temp, criterion, use_cuda = False):

		super().__init__()
		self.batch_size = batch_size
		self.temp = temp
		self.use_cuda = use_cuda

		self.positive_mask = self._get_positive_mask()
		self.sim_func = torch.nn.CosineSimilarity(dim = -1)
		self.criterion = criterion


	def _get_cosine_sim(self, x):
		return 1 - cosine_pairwise(x.float().unsqueeze(0))


	def _get_positive_mask(self):

		import numpy as np
		diag = np.eye(2 * self.batch_size)
		lower_diag = np.eye(2 * self.batch_size, k = -self.batch_size)
		upper_diag = np.eye(2 * self.batch_size, k =  self.batch_size)

		mask = torch.from_numpy((diag + lower_diag + upper_diag))
		mask = (1 - mask).type(torch.bool)

		if self.use_cuda:
			mask.cuda()


	def forward(self, zis, zjs):

		projections = torch.cat([zjs, zis], dim = 0)

		sim_matrix = self._get_cosine_sim(projections)
		sim_matrix = sim_matrix.squeeze()

		lower_pos = torch.diag(sim_matrix, -self.batch_size)
		upper_pos = torch.diag(sim_matrix, self.batch_size)

		positives = torch.cat([upper_pos, lower_pos]).view(2 * self.batch_size, 1)
		negatives = sim_matrix[self.positive_mask].view(2 * self.batch_size, -1)

		logits = torch.cat([positives, negatives], dim = 1)
		logits /= self.temp

		labels = torch.zeros(2 * self.batch_size, dtype=torch.long)
		if self.use_cuda:
			labels = labels.cuda()
			logits = logits.cuda()

		loss  = self.criterion(logits, labels)
		loss /= (2 * self.batch_size)

		return loss
'''