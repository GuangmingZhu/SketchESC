import argparse
import time
def parse_opt():

    parser = argparse.ArgumentParser()
    t=time.localtime()

    def str_to_bool(value):
        if value.lower() in {'false', 'f', '0', 'no', 'n'}:
            return False
        elif value.lower() in {'true', 't', '1', 'yes', 'y'}:
            return True
        raise ValueError(f'{value} is not a valid boolean value')

    # Overall settings

    parser.add_argument(
        '--home', '-home',
        type=str,
        default='./')

    parser.add_argument(
        '--ckpt_folder', '-ckpt',
        type=str,
        default='checkpoint/'+str(t.tm_mon)+"-"+str(t.tm_mday)+"-"+str(t.tm_hour)+"-"+str(t.tm_min)+"-"+str(t.tm_sec)+"/")

    parser.add_argument(
        '--log_folder', '-log',
        type=str,
        default='log/'+str(t.tm_mon)+"-"+str(t.tm_mday)+"-"+str(t.tm_hour)+"-"+str(t.tm_min)+"-"+str(t.tm_sec)+"/")

    parser.add_argument(
        '--dataset_path', '-dataset_path',
        type=str,
        default='./data')

    parser.add_argument(
        '--pretrain_path', '-pretrain_path',
        type=str,
        default="./pretrained_model")

    parser.add_argument(
        '--lr', '-lr',
        type=float,
        default=3e-4)

    parser.add_argument(
        '--weight_decay', '-weight_decay',
        type=float,
        default=1e-4
    )

    parser.add_argument(
        '--bs', '-bs',
        type=int,
        default=128)

    parser.add_argument(
        '--max_stroke', '-max_stroke',
        type=int,
        default=43)

    parser.add_argument(
        '--mask', '-mask',
        type=str_to_bool, nargs='?', const=True,
        default=False)

    parser.add_argument(
        '--shape_emb', '-shape_emb',
        type=str,
        default='sum')

    parser.add_argument(
        '--shape_extractor', '-shape_extractor',
        type=str,
        default='lstm')

    parser.add_argument(
        '--shape_extractor_layer', '-shape_extractor_layer',
        type=int,
        default=2)

    parser.add_argument(
        '--embedding_dropout', '-embedding_dropout',
        type=float,
        default=0
    )

    parser.add_argument(
        '--attention_dropout', '-attention_dropout',
        type=float,
        default=0
    )

    parser.add_argument(
        '--local_rank', '-local_rank',
        type=int,
        default=0)

    args = parser.parse_args()

    return args
