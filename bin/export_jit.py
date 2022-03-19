from __future__ import print_function

import argparse
import os

import torch
import yaml

from tryasr_.transformer.asr_model import init_asr_model
from tryasr_.utils.checkpoint import load_checkpoint


def get_args():
    parser = argparse.ArgumentParser(description='export your script model')
    parser.add_argument('--config', required=True, help='config file')
    parser.add_argument('--checkpoint', required=True, help='checkpoint model')
    parser.add_argument('--output_file', required=True, help='output file')
    parser.add_argument('--output_quant_file',
                        default=None,
                        help='output quantized model file')
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    # No need gpu for model export
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    with open(args.config, 'r') as fin:
        configs = yaml.load(fin, Loader=yaml.FullLoader)
    model = init_asr_model(configs)
    print(model)

    load_checkpoint(model, args.checkpoint)
    # Export jit torch script model

    script_model = torch.jit.script(model)
    script_model.save(args.output_file)
    print('Export model successfully, see {}'.format(args.output_file))

    # Export quantized jit torch script model
    if args.output_quant_file:
        quantized_model = torch.quantization.quantize_dynamic(
            model, {torch.nn.Linear}, dtype=torch.qint8
        )
        print(quantized_model)
        script_quant_model = torch.jit.script(quantized_model)
        script_quant_model.save(args.output_quant_file)
        print('Export quantized model successfully, '
              'see {}'.format(args.output_quant_file))


if __name__ == '__main__':
    main()
