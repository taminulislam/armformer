
from mmseg.registry import DATASETS
from mmseg.datasets import BaseSegDataset
 
 
@DATASETS.register_module()
class Thermal_RGBDataset(BaseSegDataset):
    """Thermal RGB Dataset for weapon detection.

    This dataset is designed for thermal and RGB image segmentation
    for weapon detection tasks. It includes multiple weapon classes
    along with human and background categories.
    """
    METAINFO = dict(
        classes=('background', 'handgun', 'human', 'knife', 'rifle', 'revolver'),
        palette=[
                    [0, 0, 0],        # background - black
                    [255, 99, 71],    # handgun - tomato  
                    [35, 97, 144],    # human - blue
                    [205, 50, 50],    # knife - red
                    [255, 20, 147],   # rifle - deep pink
                    [153, 50, 205]    # revolver - purple
                ]
    )
    def __init__(self,
                 img_suffix='.jpg',
                 seg_map_suffix='.png',
                 reduce_zero_label=False,
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix, seg_map_suffix=seg_map_suffix, reduce_zero_label=reduce_zero_label, **kwargs)
 
 

