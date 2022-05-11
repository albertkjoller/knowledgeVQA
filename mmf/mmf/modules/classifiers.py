
from mmf.modules.layers import *




# qlarifais
class Classifier(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.type == "simple":
            self.module = SimpleClassifier(config.params)
        elif config.type == "numberbatch":
            # todo
            self.module = NumberbatchClassifier(config.params)
        else:
            raise NotImplementedError("Unknown classifier type: %s" % config.type)

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


class SimpleClassifier(nn.Module):
    def __init__(self, config):
        super().__init__()
        norm_layer = get_norm(config.norm)
        act_layer = get_act(config.act)
        # defining layers
        # initial input layer
        dims = [int(config.in_dim)]
        # defining extra layers if defined in config
        for i in range(int(config.num_non_linear_layers)-1):
            dims.append(int(config.h_dim))
        # final layer
        dims.append(int(config.out_dim))
        # FCNet initialized the whole classifier
        self.main = FCNet(dims, dropout=int(config.dropout), norm=config.norm, act=config.act)

        # without activation to learn the negative values
        self.final = nn.Linear(int(config.out_dim), int(config.out_dim))


    def forward(self, input):
        logits = self.final(self.main(input))
        #self.classifier(input)
        return logits










