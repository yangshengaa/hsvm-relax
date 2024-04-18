"""
name each experiments
"""


def name_exp(args):
    """provide name tags to each experiments"""
    name = args.data
    # add data tags
    if args.data == "gaussian":
        name += f"_N{args.N}_d{args.dim}_s{args.scale}_K{args.K}"

    # add model tags
    name += f"_m{args.model}_C{args.C}"

    if args.model.lower() == "moment":
        name += f"_kappa{args.kappa}"
    elif args.model.lower() == "gd":
        name += f"_lr{args.lr}"

    # add random seed
    name += f"_seed{args.seed}"

    return name
