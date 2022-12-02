from dataclasses import dataclass


@dataclass
class GenotypingConfig:
    avg_coverage: int = 15
    use_naive_priors: bool = 15
    n_threads: int = 4
    ignore_helper_model: bool = False
    ignore_helper_variants: bool = False


    @classmethod
    def from_command_line_args(cls, args):
        return cls(
            args.average_coverage,
            args.use_naive_priors,
            args.n_threads,
            args.ignore_helper_model,
            args.ignore_helper_variants
        )