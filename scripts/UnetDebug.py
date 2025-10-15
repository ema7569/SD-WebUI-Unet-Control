from scripts.Shared import get_current_unet_blocks

class UnetDebug:
    @staticmethod
    def _trim(text, max_len=30):
        if text is None:
            return ""
        text = text.strip()
        return text[:max_len-3] + "..." if len(text) > max_len else text

    @staticmethod
    def display(cond_prompts, uncond_prompts):
        """
        Affiche les prompts positifs et négatifs par bloc UNet
        """
        print("="*100)
        print("UNet Prompts Distribution")
        print("-"*100)
        print(f"{'BLOCK':<5}|{'NAME':<6}|{'ATTN':<5}|{'COND':<32}|{'UNCOND':<32}")
        print("-"*100)

        unet_block_info = get_current_unet_blocks()

        num_blocks = min(len(cond_prompts), len(uncond_prompts), len(unet_block_info))
        
        for i in range(num_blocks):
            block = unet_block_info[i]
            name = block['name']
            attn_str = "YES" if block['attn'] else "NO"
            cond_text = UnetDebug._trim(cond_prompts[i])
            uncond_text = UnetDebug._trim(uncond_prompts[i])
            print(f"{i:02d}   |{name:<6}|{attn_str:<5}|{cond_text:<32}|{uncond_text:<32}")

        print("-"*100)
        print("")