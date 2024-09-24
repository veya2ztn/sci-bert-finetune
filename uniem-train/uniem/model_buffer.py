from enum import Enum
from pathlib import Path
from typing import ClassVar, Literal, Protocol, Type, TypeVar, cast

import numpy as np
import torch

from uniem.criteria import (
    ContrastLoss,
    PairInBatchNegCoSentLoss,
    PairInBatchNegSigmoidContrastLoss,
    PairInBatchNegSoftmaxContrastLoss,
)

from uniem.model import Embedder, EmbedderForTrain, InBatchNegLossType



class EmbedderForPairInBatchNegTrain_buffer(EmbedderForTrain):
    def __init__(
        self,
        embedder: Embedder,
        temperature: float = 0.05,
        buffer_num: int = 3,
        loss_type: InBatchNegLossType | str = InBatchNegLossType.softmax,
    ):
        self.buffer_num = buffer_num
        self.loss_type = InBatchNegLossType(loss_type)

        match self.loss_type:
            case InBatchNegLossType.sigmoid:
                criterion = PairInBatchNegSigmoidContrastLoss(temperature)
            case InBatchNegLossType.softmax:
                criterion = PairInBatchNegSoftmaxContrastLoss(temperature)
            case InBatchNegLossType.cosent:
                criterion = PairInBatchNegCoSentLoss(temperature)
        super().__init__(embedder, criterion)
        self.register_buffer('buffer', None)
        self.register_buffer('pos_buffer', None)

    def forward(self, text_ids: torch.Tensor, text_pos_ids: torch.Tensor) -> dict[str, torch.Tensor]:
        text_embeddings = self.embedder(text_ids)
        text_pos_embeddings = self.embedder(text_pos_ids)

        bt_sz = text_embeddings.size(0)

        if self.buffer != None: 
            text_embeddings = torch.cat([self.buffer[-bt_sz * (self.buffer_num-1):], text_embeddings], dim=0)
            text_pos_embeddings = torch.cat([self.pos_buffer[-bt_sz * (self.buffer_num-1):], text_pos_embeddings], dim=0)
            del self.buffer
            del self.pos_buffer

        
        loss = self.criterion(text_embeddings, text_pos_embeddings)

        self.buffer = text_embeddings.detach()
        self.pos_buffer = text_pos_embeddings.detach()

        return {'loss': loss}
  