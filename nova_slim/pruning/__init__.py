from .finegrained_pruning import *
from .structured_pruning import *
from .apply_compression import *
from .one_shot import *
from .agp import *
from .lottery_ticket import LotteryTicketPruner


PRUNER_DICT = {
    "slim": SlimPruner,
    "l1": L1FilterPruner,
    "l2": L2FilterPruner,
    "fpgm": FPGMPruner,
    "taylor": TaylorFOWeightFilterPruner,
    "act_apoz": ActivationAPoZRankFilterPruner,
    "act_mean": ActivationMeanRankFilterPruner
}