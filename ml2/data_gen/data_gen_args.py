"""Common data generation arguments"""


def add_data_gen_frac_args(parser):
    parser.add_argument("--train-frac", type=float, default=0.8, metavar="fraction")
    parser.add_argument("--val-frac", type=float, default=0.1, metavar="fraction")
    parser.add_argument("--test-frac", type=float, default=0.1, metavar="fraction")


def add_data_gen_args(parser):
    add_data_gen_frac_args(parser)
    parser.add_argument(
        "--add-to-wandb", action="store_true", help="add data to Weights and Biases"
    )
    parser.add_argument("-n", "--num-samples", type=int, default=100, help="number of samples")
    parser.add_argument("--name", type=str, metavar="NAME", required=True, help="dataset name")
    parser.add_argument("--project", type=str, metavar="PROJECT", help="dataset project")
    parser.add_argument(
        "-u", "--upload", action="store_true", help="upload generated data to GCP storage bucket"
    )


def add_dist_data_gen_args(parser):
    add_data_gen_args(parser)
    parser.add_argument(
        "--batch-size", type=int, default=10, help="size of batches provided to worker"
    )
    parser.add_argument("--num-workers", type=int, default=4, help="number of workers")
