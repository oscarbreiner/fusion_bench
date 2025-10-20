from torch import nn
from fusion_bench.method import BaseAlgorithm
from fusion_bench.modelpool import BaseModelPool

class ReturnFirstModelAlgorithm(BaseAlgorithm):
    """Return exactly the first non-'_pretrained_' model in the pool."""
    def run(self, modelpool: BaseModelPool):
        if isinstance(modelpool, nn.Module):
            return modelpool
        if not isinstance(modelpool, BaseModelPool):
            modelpool = BaseModelPool(modelpool)
        names = [n for n in modelpool.model_names if n != "_pretrained_"]
        assert names, "No non-pretrained model provided in modelpool."
        return modelpool.load_model(names[0])