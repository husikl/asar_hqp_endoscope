import sys
from argparse import ArgumentParser


def _get_config():
    parser = ArgumentParser()
    parser.add_argument('--model', default='./weights/XMem.pth')
    parser.add_argument('--buffer_size', help='Correlate with CPU memory consumption', type=int, default=100)
    parser.add_argument('--num_objects', type=int, default=1)
    
    # Long-memory options
    # Defaults. Some can be changed in the GUI.
    parser.add_argument('--max_mid_term_frames', help='T_max in paper, decrease to save memory', type=int, default=10)
    parser.add_argument('--min_mid_term_frames', help='T_min in paper, decrease to save memory', type=int, default=5)
    parser.add_argument('--max_long_term_elements', help='LT_max in paper, increase if objects disappear for a long time', 
                                                    type=int, default=10000)
    parser.add_argument('--num_prototypes', help='P in paper', type=int, default=128) 

    parser.add_argument('--top_k', type=int, default=30)
    parser.add_argument('--mem_every', type=int, default=10)
    parser.add_argument('--deep_update_every', help='Leave -1 normally to synchronize with mem_every', type=int, default=-1)
    parser.add_argument('--no_amp', help='Turn off AMP', action='store_true')
    parser.add_argument('--size', default=480, type=int, 
            help='Resize the shorter side to this size. -1 to use original resolution. 480 for default')
    args = parser.parse_args()
    
    config = vars(args)
    config['enable_long_term'] = True
    config['enable_long_term_count_usage'] = True
    
    return config
