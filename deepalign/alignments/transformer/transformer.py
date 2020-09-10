from deepalign.alignments.transformer.core import Transformer


class TransformerModel(Transformer):
    version = 1
    abbreviation = 'TR'
    name = 'Transformer'
    config = dict(num_encoder_layers=2, mha_heads=6, dropout_rate=0.1, ff_dim=32, fixed_emb_dim=30)

    def __init__(self, dataset, **ad_kwargs):
        super(TransformerModel, self).__init__(dataset, **ad_kwargs, **self.config)
