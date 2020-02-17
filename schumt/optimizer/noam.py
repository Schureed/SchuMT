import torch.optim as optim

from . import builder
from .optimzer import Optimizer


@builder.register("noam")
class NoamOptimizer(Optimizer):
    def __init__(self, parameters, d_model):
        def inverse_sqrt(step, warmup, scale):
            if step < warmup:
                return scale * step * (warmup ** -1.5)
            else:
                return scale * (step ** -0.5)

        super(NoamOptimizer, self).__init__(parameters)
        self.optimizer = optim.Adam(
            parameters,
            lr=self.lr,
            betas=(self.args['beta1'], self.args['beta2']),
            eps=(self.args['eps'])
        )

        self.scheduler = optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lambda s: inverse_sqrt(step=s, warmup=self.args['warmup_step'], scale=d_model ** -0.5)
        )

    @classmethod
    def register_args(cls, parser, source) -> dict:
        parser.add_argument("--beta1", type=float, default=0.9)
        parser.add_argument("--beta2", type=float, default=0.98)
        parser.add_argument("--eps", type=float, default=1e-9)
        parser.add_argument("--warmup-step", type=int, default=4000)
        args, _ = parser.parse_known_args(source)
        return args.__dict__

    def _clear_grad(self):
        self.optimizer.zero_grad()

    def _update_parameters(self):
        self.optimizer.step()
        self.scheduler.step()
