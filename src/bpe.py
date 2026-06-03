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
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"

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
        self.vocab_size = vocab_size
        self.id_to_token = {}
        self.token_to_id = {}
        self.merges = []

    def _init_special_tokens(self):
        # 1.특수 토큰 4개를 고정 ID 0~3에 등록
        for token, id in SPECIAL_IDS.items():
            self.id_to_token[id] = token
            self.token_to_id[token] = id

        # 2. byte 0~255를 ID 4~259에 bytes([byte_value]) 형태로 등록.
        for byte_val in range(NUM_BYTES):
            token = bytes([byte_val])
            idx = BYTE_OFFSET + byte_val
            self.id_to_token[idx] = token
            self.token_to_id[token] = idx

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

    """ 코퍼스에서 BPE merge rule과 vocabulary를 학습 """

    def train(self, corpus: str):
        # 1. 이전 학습 상태 초기화 후 특수 토큰 등록
        self.id_to_token = {}
        self.token_to_id = {}
        self.merges = []
        self._init_special_tokens()
        ids = [BYTE_OFFSET + byte for byte in corpus.encode("utf-8")]

        # vocab이 찰 때까지 반복
        while len(self.id_to_token) < self.vocab_size:
            # 2. 인접 쌍 빈도수 카운팅
            pair_counts = {}
            for i in range(len(ids) - 1):
                pair = (ids[i], ids[i + 1])
                pair_counts[pair] = pair_counts.get(pair, 0) + 1

            if not pair_counts:
                break

            # 3. 가장 빈도 높은 쌍 선택
            best_pair = max(pair_counts, key=lambda p: pair_counts[p])
            if pair_counts[best_pair] < 2:
                break

            # 4. 새 토큰 등록
            new_id = len(self.id_to_token)
            self.id_to_token[new_id] = best_pair
            self.token_to_id[best_pair] = new_id
            self.merges.append(best_pair)

            # 5. 리스트에서 best_pair를 new_id로 교체
            i = 0
            merged = []
            while i < len(ids):
                if i < len(ids) - 1 and (ids[i], ids[i + 1]) == best_pair:
                    merged.append(new_id)
                    i += 2
                else:
                    merged.append(ids[i])
                    i += 1
            ids = merged

    # 학습된 vocab과 merges를 JOSN 파일로 저장
    # 입력 : 파일 경로
    # 출력(결과) : 파일 생성
    def save(self, path: str | Path):
        """
        TODO: vocabulary와 merge rule을 JSON 파일로 저장합니다.

        bytes와 tuple은 JSON에 바로 저장할 수 없으므로 type 정보를 함께 저장하세요.
        """
        vocab = {}
        for id, token in self.id_to_token.items():
            if isinstance(token, bytes):
                vocab[id] = {"type": "bytes", "value": list(token)}
            elif isinstance(token, tuple):
                vocab[id] = {"type": "tuple", "value": list(token)}
            else:
                vocab[id] = {"type": "str", "value": token}

        merges = [list(pair) for pair in self.merges]

        with open(path, "w", encoding="utf-8") as f:
            json.dump({"vocab_size": self.vocab_size, "id_to_token": vocab, "merges": merges}, f)

    # 저장된 JSON를 읽어서 vocab과 merges를 복원
    # 입력 : 파일 경로
    # 출력(결과) : 저장된 JSON을 읽어서 vocab과 merges를 복원
    def load(self, path: str | Path):
        """
        TODO: save()로 저장한 JSON 파일을 읽어 vocabulary와 merge rule을 복원합니다.
        """
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.vocab_size = data["vocab_size"]
        self.id_to_token = {}
        self.token_to_id = {}

        for id_str, entry in data["id_to_token"].items():
            id = int(id_str)
            if entry["type"] == "bytes":
                token = bytes(entry["value"])
            elif entry["type"] == "tuple":
                token = tuple(entry["value"])
            else:
                token = entry["value"]
            self.id_to_token[id] = token
            self.token_to_id[token] = id

        self.merges = [tuple(pair) for pair in data["merges"]]

    """ 문자열을 token ID 리스트로 변환 """

    def encode(self, text: str, add_bos_eos: bool = False) -> list[int]:
        if not self.id_to_token:
            self._init_special_tokens()
        # 1. text를 byte ID로 전환
        byte_ids = [BYTE_OFFSET + byte for byte in text.encode("utf-8")]

        # 2. 3개로 쪼개져있는 byte를 merge rules 순서대로 적용
        # merges에는 합쳐야 되는 두 byte가 tuple 형태로 저장되어 있다. (순서대로)
        for pair in self.merges:
            merged_id = self.token_to_id[pair]
            i = 0
            merged_result = []
            while i < len(byte_ids):
                # 만약 merges rule에 존재하다면, merged 배열에 합쳐진 id를 추가
                if i < len(byte_ids) - 1 and (byte_ids[i], byte_ids[i + 1]) == pair:
                    merged_result.append(merged_id)
                    i += 2
                # merges rule에 존재하지 않는다면, merged 배열에 현재 byte ID를 추가
                else:
                    merged_result.append(byte_ids[i])
                    i += 1
            byte_ids = merged_result

        # 3. add_bos_eos가 True라면, 앞뒤에 bos/eos ID 추가
        if add_bos_eos:
            byte_ids = [self.get_bos_id()] + byte_ids + [self.get_eos_id()]

        return byte_ids

    """ token ID 리스트를 문자열로 복원 """

    def decode(self, ids: list[int], skip_special: bool = True) -> str:
        byte_list = []

        # ID 하나를 bytes로 풀어주는 헬퍼 함수
        def devide_ID(id):
            token = self.id_to_token[id]
            # id_to_token 배열에 토플 형태로 저장되어 있으면, 재귀적으로 분해
            if isinstance(token, tuple):
                devide_ID(token[0])
                devide_ID(token[1])
            # bytes라면, 그대로 byte_list에 추가
            elif isinstance(token, bytes):
                byte_list.append(token[0])
            # tuple, bytes 둘 다 아니라면 -> 특수 토큰
            else:
                # skip_special이 false라면 특수 토큰도 byte_list에 추가
                if not skip_special:
                    byte_list.extend(token.encode("utf-8"))

        # id 리스트를 순회하면서 devide_ID 호출
        for id in ids:
            if id in self.id_to_token:
                devide_ID(id)

        return bytes(byte_list).decode("utf-8")
