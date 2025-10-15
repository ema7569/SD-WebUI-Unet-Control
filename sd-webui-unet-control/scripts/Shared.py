NUM_BLOCKS = 27  # 12 input, 3 middle, 12 output

# --- Structure de référence du UNet ---
# U-Net Control for SD1.5 (27 blocks: 12 IN, 3 M, 12 OUT)
#---------------------------------------------------------

#these structure have info about unet block, 
#the attn field is used to skip distribution when prompt attempt to 
#write on non-unet. This if because it's possible to have bug when writing on
#non-unet blocks.

#note, unet_block_info 
#M00-M02 are open (true) because they seem to have influence on results

CURRENT_UNET_MODEL = "sd15"   # current model 
CURRENT_UNET_TYPE = "unet"    # "unet" or "free"

UNET_BLOCKS = {
    "sd15": {
        "unet": [
            {'index': 0, 'name': 'IN00', 'attn': False},
            {'index': 1, 'name': 'IN01', 'attn': True},
            {'index': 2, 'name': 'IN02', 'attn': True},
            {'index': 3, 'name': 'IN03', 'attn': False},
            {'index': 4, 'name': 'IN04', 'attn': True},
            {'index': 5, 'name': 'IN05', 'attn': True},
            {'index': 6, 'name': 'IN06', 'attn': False},
            {'index': 7, 'name': 'IN07', 'attn': True},
            {'index': 8, 'name': 'IN08', 'attn': True},
            {'index': 9, 'name': 'IN09', 'attn': False},
            {'index': 10, 'name': 'IN10', 'attn': False},
            {'index': 11, 'name': 'IN11', 'attn': False},
            {'index': 12, 'name': 'M00', 'attn': True},
            {'index': 13, 'name': 'M01', 'attn': True},
            {'index': 14, 'name': 'M02', 'attn': True},
            {'index': 15, 'name': 'OUT00', 'attn': False},
            {'index': 16, 'name': 'OUT01', 'attn': False},
            {'index': 17, 'name': 'OUT02', 'attn': False},
            {'index': 18, 'name': 'OUT03', 'attn': True},
            {'index': 19, 'name': 'OUT04', 'attn': True},
            {'index': 20, 'name': 'OUT05', 'attn': True},
            {'index': 21, 'name': 'OUT06', 'attn': True},
            {'index': 22, 'name': 'OUT07', 'attn': True},
            {'index': 23, 'name': 'OUT08', 'attn': True},
            {'index': 24, 'name': 'OUT09', 'attn': True},
            {'index': 25, 'name': 'OUT10', 'attn': True},
            {'index': 26, 'name': 'OUT11', 'attn': True},
        ],

        "free": [
            {'index': 0, 'name': 'IN00', 'attn': True},
            {'index': 1, 'name': 'IN01', 'attn': True},
            {'index': 2, 'name': 'IN02', 'attn': True},
            {'index': 3, 'name': 'IN03', 'attn': True},
            {'index': 4, 'name': 'IN04', 'attn': True},
            {'index': 5, 'name': 'IN05', 'attn': True},
            {'index': 6, 'name': 'IN06', 'attn': True},
            {'index': 7, 'name': 'IN07', 'attn': True},
            {'index': 8, 'name': 'IN08', 'attn': True},
            {'index': 9, 'name': 'IN09', 'attn': True},
            {'index': 10, 'name': 'IN10', 'attn': True},
            {'index': 11, 'name': 'IN11', 'attn': True},
            {'index': 12, 'name': 'M00', 'attn': True},
            {'index': 13, 'name': 'M01', 'attn': True},
            {'index': 14, 'name': 'M02', 'attn': True},
            {'index': 15, 'name': 'OUT00', 'attn': True},
            {'index': 16, 'name': 'OUT01', 'attn': True},
            {'index': 17, 'name': 'OUT02', 'attn': True},
            {'index': 18, 'name': 'OUT03', 'attn': True},
            {'index': 19, 'name': 'OUT04', 'attn': True},
            {'index': 20, 'name': 'OUT05', 'attn': True},
            {'index': 21, 'name': 'OUT06', 'attn': True},
            {'index': 22, 'name': 'OUT07', 'attn': True},
            {'index': 23, 'name': 'OUT08', 'attn': True},
            {'index': 24, 'name': 'OUT09', 'attn': True},
            {'index': 25, 'name': 'OUT10', 'attn': True},
            {'index': 26, 'name': 'OUT11', 'attn': True},
        ]
    },

    # Préparation d’un modèle futur
    "sdxl": {
        "unet": [],
        "free": [],
    }
}

def get_current_unet_blocks():
    return UNET_BLOCKS[CURRENT_UNET_MODEL][CURRENT_UNET_TYPE]

def set_current_unet_blocks(unet_type = "unet"):
    global CURRENT_UNET_TYPE
    CURRENT_UNET_TYPE = unet_type

