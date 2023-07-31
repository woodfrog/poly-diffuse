import argparse


def get_args_parser():
    parser = argparse.ArgumentParser('Holistic edge attention transformer', add_help=False)
    parser.add_argument('--data_path', default='',
                        help='path to the data`')
    parser.add_argument('--lr', default=2e-4, type=float)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--weight_decay', default=1e-5, type=float)
    parser.add_argument('--epochs', default=800, type=int)
    parser.add_argument('--lr_drop', default=600, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')
    parser.add_argument('--print_freq', default=40, type=int)
    parser.add_argument('--output_dir', default='./checkpoints/ckpts_s3d_256',
                        help='path where to save, empty for no saving')
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--image_size', default=256, type=int)
    parser.add_argument('--run_validation', action='store_true',
            help='Whether run validation or not, default: False')
    parser.add_argument('--distributed', action='store_true', help='Run distributed training')
    parser.add_argument('--hidden_dim', default=256, type=int, help='hidden dim of the core transformer-based model')
    parser.add_argument('--repeat_train', default=1, type=int, help='Repeat the training set for each epoch')
    return parser
