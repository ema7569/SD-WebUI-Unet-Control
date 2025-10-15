import re

from scripts.Shared import get_current_unet_blocks

class UnetParser:
    """
    Parser U-Net Control prompts avec syntaxe avancée :

    - Blocs simples : &I00="text", &I05:07="text", &I00-10="text"
    - Double branche : &I00+="text", &I00-08+="text", &I00:05:06+="text"
    - Combinaison IN/OUT : &I04-07&O08-11="text", &I06:08&O01:05:07="text"
    - Blocs OUT seuls : &O01:05:07="text"
    
    Le texte global (sans sélecteur) est appliqué à tous les blocs ATTN par défaut.
    """

    # Pattern pour extraire sélecteurs et texte
    # Exemple : &I00-11="text" ou &I00+ = "text" ou &I00-08&O01-05="text"
    pattern = re.compile(r'&([IMO])([\d:\-]+)(\+?)(?:&O([\d:\-]+))?="([^"]*)"')

    def __init__(self):
        self.warnings = []

    def parse(self, prompt: str):
        """
        Retourne une liste de 27 prompts fusionnés selon unet_block_info.
        Les blocs non-ATTN restent "".
        """
        unet_block_info = get_current_unet_blocks()
        
        blocks = [""] * len(unet_block_info)
        base_prompt = prompt
        matches = list(self.pattern.finditer(prompt))

        for m in matches:
            block_type = m.group(1)       # I, M, O
            selector_in = m.group(2)      # ex: 00-08 ou 00:05:06
            plus_flag = bool(m.group(3))  # True si "+"
            selector_out = m.group(4)     # ex: 01:05:07 pour &Ixx&Oyy
            text = m.group(5).strip()     # texte à appliquer

            # Calcul des indices IN
            in_indices = self._parse_selector(selector_in, block_type)

            # Calcul des indices OUT
            out_indices = []
            if plus_flag and block_type == "I" and selector_out is None:
                # propagation IN → OUT alignée
                out_indices = self._map_in_to_out(in_indices)
            elif selector_out:
                out_indices = self._parse_selector(selector_out, "O")

            # Appliquer texte sur IN
            for i in in_indices:
                if unet_block_info[i]['attn']:
                    blocks[i] = self._concat(blocks[i], text)

            # Appliquer texte sur OUT
            for j in out_indices:
                if unet_block_info[j]['attn']:
                    blocks[j] = self._concat(blocks[j], text)

            # Retirer le match du prompt global
            base_prompt = base_prompt.replace(m.group(0), "")

        # Appliquer le prompt global restant sur tous les blocs ATTN
        base_prompt = base_prompt.strip()
        if base_prompt:
            for i, info in enumerate(unet_block_info):
                if info['attn']:
                    blocks[i] = self._concat(blocks[i], base_prompt)

        return blocks

    def _parse_selector(self, selector: str, block_type: str):
        """
        Retourne la liste d'indices globaux correspondant à &I/M/O.
        Gère blocs uniques, plages (-), listes (:)
        """
        if block_type == "I":
            base = 0
            max_local = 11
        elif block_type == "M":
            base = 12
            max_local = 2
        else:  # O
            base = 15
            max_local = 11

        blocks = set()
        selector = selector.strip()
        if not selector:
            return []

        # plage "start-end"
        if "-" in selector:
            parts = selector.split("-")
            if len(parts) != 2:
                raise ValueError(f"Plage invalide: {selector}")
            start, end = map(int, parts)
            if start < 0 or end < 0 or start > end or end > max_local:
                raise ValueError(f"Index hors limites pour bloc {block_type}")
            blocks.update(range(base + start, base + end + 1))
        # liste "n1:n2:n3"
        elif ":" in selector:
            for part in selector.split(":"):
                part = part.strip()
                if not part:
                    continue
                n = int(part)
                if n < 0 or n > max_local:
                    raise ValueError(f"Index hors limites pour bloc {block_type}")
                blocks.add(base + n)
        # bloc unique
        else:
            n = int(selector)
            if n < 0 or n > max_local:
                raise ValueError(f"Index hors limites pour bloc {block_type}")
            blocks.add(base + n)

        return sorted(blocks)

    def _map_in_to_out(self, in_indices):
        """
        Mapping proportionnel IN -> OUT.
        - Aligne les indices en fonction de leur position relative dans la plage 0–11.
        - Bloc unique : map direct proportionnellement à OUT.
        """
        out_base = 15
        out_max = 11  # 0–11
        n_out = out_max + 1

        if not in_indices:
            return []

        mapped_out = []
        for i in in_indices:
            local_in = i  # 0–11
            if local_in < 0:
                continue
            # proportion dans l’échelle IN → OUT
            ratio = local_in / 11
            mapped_local_out = round(ratio * out_max)
            mapped_out.append(out_base + mapped_local_out)

        return sorted(set(mapped_out))


    @staticmethod
    def _concat(existing, new_text):
        if existing:
            return existing + " " + new_text
        return new_text
