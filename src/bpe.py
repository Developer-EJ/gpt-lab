# -*- coding: utf-8 -*-
"""
UTF-8 byte-level BPE 토크나이저 과제 템플릿.

외부 tokenizer 라이브러리 없이 BPE(Byte Pair Encoding)를 직접 구현합니다.
한국어 NSMC 리뷰를 다루므로 문자열을 글자/공백 단위로 먼저 자르지 말고,
항상 `text.encode("utf-8")`로 byte ID 시퀀스를 만든 뒤 merge를 적용하세요.
"""

from pathlib import Path
import json


PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
BOS_TOKEN = "<bos>" # 문장 시작 토큰
EOS_TOKEN = "<eos>" # 문장 끝 토큰

SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN]
SPECIAL_IDS = {token: idx for idx, token in enumerate(SPECIAL_TOKENS)}
BYTE_OFFSET = len(SPECIAL_TOKENS)
NUM_BYTES = 256


class BPETokenizer:
    """
    UTF-8 byte-level BPE 토크나이저.

    권장 ID 배치:
    - 0~3: <pad>, <unk>, <bos>, <eos>
    - 4~259: 원본 byte 0~255
    - 260 이상: BPE merge로 생성한 토큰
    """

    def __init__(self, vocab_size: int = 3000):
        # 학습할 vocabulary 크기와 token ID 변환 사전을 준비합니다.
        self.vocab_size = vocab_size
        self.id_to_token = {} # token ID로 실제 token을 찾는 사전
        self.token_to_id = {} # 실제 token으로 token ID를 찾는 사전
        self.merges = [] # 학습된 merge 규칙을 순서대로 저장하는 리스트

    def _init_special_tokens(self):
        """
        TODO:
        1. 특수 토큰 4개를 고정 ID 0~3에 등록합니다.
        2. byte 0~255를 ID 4~259에 bytes([byte_value]) 형태로 등록합니다.
        """
        for idx, token in enumerate(SPECIAL_TOKENS):
            self.id_to_token[idx] = token
            self.token_to_id[token] = idx

        for byte_value in range(NUM_BYTES): # 0 ~ 255
            token_id = byte_value + BYTE_OFFSET # 4 ~ 259
            self.id_to_token[token_id] = bytes([byte_value])
            self.token_to_id[bytes([byte_value])] = token_id

    def get_pad_id(self):
        """padding 토큰 ID."""
        return SPECIAL_IDS[PAD_TOKEN]

    def get_unk_id(self):
        """unknown 토큰 ID."""
        return SPECIAL_IDS[UNK_TOKEN]

    def get_bos_id(self):
        """문장 시작 토큰 ID."""
        return SPECIAL_IDS[BOS_TOKEN]

    def get_eos_id(self):
        """문장 끝 토큰 ID."""
        return SPECIAL_IDS[EOS_TOKEN]

    def train(self, corpus: str):
        """
        TODO: 코퍼스에서 BPE merge rule과 vocabulary를 학습합니다.

        구현 힌트:
        - `corpus.encode("utf-8")`로 byte ID 시퀀스를 만듭니다.
        - 가장 자주 등장하는 이웃 token pair를 찾습니다.
        - 새 token ID를 만들고, 시퀀스의 해당 pair를 새 ID로 치환합니다.
        - `self.merges`, `self.id_to_token`, `self.token_to_id`를 갱신합니다.
        """
        # 기본 vocab을 준비하고 corpus를 byte token ID 시퀀스로 바꿉니다.
        self._init_special_tokens()
        ids = [byte_value + BYTE_OFFSET for byte_value in corpus.encode("utf-8")]

        while len(self.id_to_token) < self.vocab_size:
            # 인접한 token pair의 등장 횟수를 셉니다.
            counts = {}
            for i in range(len(ids) - 1):
                pair = (ids[i], ids[i+1])
                counts[pair] = counts.get(pair, 0) + 1 # 처음 나온 pair는 0에서 시작해 1 증가
            
            if not counts: # counts가 비어있으면
                break

            best_pair = max(counts, key=lambda x: counts[x]) # 가장 많은 pair

            # 가장 자주 나온 pair를 새 merge token으로 등록합니다.
            new_id = len(self.id_to_token)
            self.id_to_token[new_id] = best_pair
            self.token_to_id[best_pair] = new_id
            self.merges.append(best_pair)

            # 시퀀스 안의 best_pair를 새 token ID로 치환합니다.
            a, b = best_pair
            result = []
            i = 0
            while i < len(ids):
                if i < len(ids) - 1 and ids[i] == a and ids[i+1] == b:
                    result.append(new_id)
                    i += 2
                else:
                    result.append(ids[i])
                    i += 1
            ids = result

    def save(self, path: str | Path):
        """
        TODO: vocabulary와 merge rule을 JSON 파일로 저장합니다.
        bytes와 tuple은 JSON에 바로 저장할 수 없으므로 type 정보를 함께 저장하세요.
        """
        # JSON은 tuple을 직접 저장하지 못하므로 merge pair를 list로 저장합니다.
        # 기본 vocab은 load에서 다시 만들 수 있어 merges만 저장합니다.
        data = {
            "vocab_size" : self.vocab_size,
            "merges" : [[a, b] for (a, b) in self.merges]
        }

        # open(파일경로, 모드, 인코딩)
        # json.dump(저장할 데이터, 파일 객체) 
        with open(Path(path), "w", encoding="utf-8") as f:
            json.dump(data, f)

    def load(self, path: str | Path):
        """
        TODO: save()로 저장한 JSON 파일을 읽어 vocabulary와 merge rule을 복원합니다.
        """
        # 저장된 merge list를 읽고, 다시 tuple pair로 복원합니다.
        with open(Path(path), "r", encoding="utf-8") as f:
            data = json.load(f)

        self.vocab_size = data["vocab_size"] # vocab_size 복원

        # 0~3 특수 토큰, 4~259 byte token을 먼저 복원합니다.
        self._init_special_tokens()

        for pair_list in data["merges"]:
            a = pair_list[0]
            b = pair_list[1]

            # merge 순서대로 새 token ID를 다시 부여합니다.
            new_id = len(self.id_to_token) # 현재 vocab 크기가 새 ID
            self.id_to_token[new_id] = (a, b)
            self.token_to_id[(a, b)] = new_id
            self.merges.append((a, b))

    def encode(self, text: str, add_bos_eos: bool = False) -> list[int]:
        """
        TODO: 문자열을 token ID 리스트로 변환합니다.

        구현 힌트:
        - 먼저 UTF-8 byte ID 리스트를 만듭니다.
        - train/load에서 얻은 merge rule을 학습 순서대로 적용합니다.
        - add_bos_eos=True이면 앞뒤에 bos/eos ID를 붙입니다.
        """
        # 문자열을 UTF-8 byte로 바꾸고, byte token ID로 변환합니다.
        ids = [byte_value + BYTE_OFFSET for byte_value in text.encode("utf-8")]

        # 학습된 merge rule을 순서대로 적용합니다.
        for (a, b) in self.merges:
            new_id = self.token_to_id[(a, b)]
            result = []
            i = 0
            while i < len(ids):
                if i < len(ids) - 1 and ids[i] == a and ids[i+1] == b:
                    result.append(new_id)
                    i += 2
                else:
                    result.append(ids[i])
                    i += 1
            ids = result
        
        # 옵션이 켜져 있으면 문장 앞뒤에 BOS/EOS를 붙입니다.
        if add_bos_eos:
            ids = [self.get_bos_id()] + ids + [self.get_eos_id()]
            
        return ids

    def decode(self, ids: list[int], skip_special: bool = True) -> str:
        """
        TODO: token ID 리스트를 문자열로 복원합니다.

        주의:
        - merge token은 원본 byte token까지 재귀적으로 펼칩니다.
        - byte를 하나씩 decode하지 말고, 마지막에 `bytes(...).decode("utf-8")`를 한 번만 호출합니다.
        """
        # token ID를 원본 byte 조각으로 되돌립니다.
        def _to_bytes(id):
            token = self.id_to_token[id]

            if isinstance(token, str): # 특수 토큰
                return b""
            if isinstance(token, bytes): # byte 토큰
                return token
            if isinstance(token, tuple): # merge 토큰
                a, b = token
                return _to_bytes(a) + _to_bytes(b) 
        
        # skip_special=True이면 특수 토큰을 건너뛰고, 나머지는 byte로 이어 붙입니다.
        all_bytes = b""
        for id in ids:
            if skip_special and id in (0, 1, 2, 3):
                continue
            all_bytes += _to_bytes(id)
        
        # UTF-8 decode는 마지막에 한 번만 수행합니다.
        return all_bytes.decode("utf-8")
