# weight_for_empty_mask=1.0
EPOCHS = 10

class Config:
    class GENERAL:
        EPOCHS = EPOCHS
        BATCH_SIZE = 16
        LR = 1e-3
        CHANNELS = 1
        NUM_CLASSES = 1
        NUM_WORKERS = 4
        OVERLAP_THRESHOLD = 0.2
        DEVICE = 'cuda:0'
        METRIC = 'v3'
        CSV_RANDOM_SEED = 821
        REMOVE_MASK = False
        EACH_MASK = True
            
    class MODEL:
        # NAME = 'UNet'
        NAME = 'MIMEViT'
        PRETRAINED = False
    
    class SCHEDULER:
        NAME = 'CosineAnnealingLR'
        T_MAX = EPOCHS
        ETA_MIN = 1e-5
        
    class OPTIMIZER:
        NAME = 'AdamW'
        IF_PCGRAD = False

    class AUGMENTATION:
        CLAHE = False
        RANDOMSIZEDCROP = False
        RESIZE = False
        BREASTSHIFT = False
        HFLIP = False
        VFLIP = True
        ELASTIC = False
        GAUSS_NOISE = False
        GRID_DISTORTION = False
        ROTATE = True
        class ROTATE:
            LIMIT = 40
        EQUALIZE = False
        UNSHARP_MASK = False
            
        MEAN = 0.5
        STD = 0.5