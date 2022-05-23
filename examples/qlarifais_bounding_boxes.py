
import sys

from mmf.models.frcnn import GeneralizedRCNN
sys.path.append("..")

if __name__ == "__main__":
    config = "/mmf/mmf/configs/other/feat_configs/R-50-grid.yaml"

    # Init FasterRCNN
    model = GeneralizedRCNN(config)




