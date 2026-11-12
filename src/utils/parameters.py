import argparse


class InferenceOnlyParameters:
    def __init__(self):
        self.opt = None
        self.parser = argparse.ArgumentParser()

        self.parser.add_argument('--checkpoint-path', help='Path of the model checkpoint', required=True, dest='checkpoint_path')
        self.parser.add_argument('--data-path', help='Path of the data to use at inference time', required=True, dest='data_path')
        self.parser.add_argument('--save-out', help='Whether to save the output tractogram (with embeddings) in trx format', required=False, dest='save', action='store_true')
        self.parser.add_argument('--out-path', help='Where to save the output of the embedding', required=False, default=None, dest='out_path')
        self.parser.add_argument('--reference', help='Reference T1 used to load the tractograms. Used only for tck format, ignored with trk or trx.', required=False, dest='reference', default='same')
        self.parser.add_argument('-y', help='Whether to overwrite the output file if it already exists',
                                 default=False, action='store_true', required=False, dest='auto_yes')
        self.parser.add_argument('-v', help='Whether to use verbose output or not', action='store_true', dest='verbose', default=False)
        self.parser.add_argument('--cuda', help='Which GPU to use for embedding', type=int, default=None, required=False, dest='cuda')
        self.parser.add_argument('--chunk-dim', help='Number of streamlines embedded at once (lower it if you run out of memory)',
                                 type=int, default=10000, required=False, dest='chunk_dim')
        self.parser.add_argument('--workers', help='Number of workers used to embed the streamlines',
                                 type=int, default=None, required=False, dest='workers')

    def parse(self):
        self.opt = self.parser.parse_args()
        return self.opt
