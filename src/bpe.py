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
            json.dump(
                {"vocab_size": self.vocab_size, "id_to_token": vocab, "merges": merges},
                f,
            )

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

    def decode(
        self,
        ids: list[int],
        skip_special: bool = True,
        errors: str = "strict",
    ) -> str:
        """
        TODO: token ID 리스트를 문자열로 복원합니다.

        주의:
        - merge token은 원본 byte token까지 재귀적으로 펼칩니다.
        - byte를 하나씩 decode하지 말고, 마지막에 `bytes(...).decode("utf-8")`를 한 번만 호출합니다.
        """

        """
        token ID 리스트를 문자열로 복원합니다.

        Input:
            ids:
                token ID 리스트입니다.
                예: [69]
                예: [238, 180, 132]
                예: [2, 69, 3]

            skip_special:
                True이면 <pad>, <unk>, <bos>, <eos> 같은 특수 토큰은 복원 결과에서 제외합니다.

        Output:
            복원된 문자열입니다.

            예:
                decode([69])
                -> "A"

            예:
                decode([238, 180, 132])
                -> "가"

            예:
                decode([2, 69, 3], skip_special=True)
                -> "A"

        중요한 점:
            - 기본 byte token은 bytes 객체입니다.
            - BPE merge token은 (left_id, right_id) 형태의 tuple입니다.
            - merge token은 바로 문자로 바꾸지 않고, 내부 token들을 재귀적으로 펼쳐 byte로 만듭니다.
            - 한글은 UTF-8에서 여러 byte로 이루어질 수 있으므로 byte 하나씩 decode하면 안 됩니다.
            - 모든 byte를 합친 뒤 마지막에 decode("utf-8")를 한 번만 호출합니다.
        """

        if not self.id_to_token:
            self._init_special_tokens()

        def token_to_bytes(token_id: int) -> bytes:
            """
            token ID 하나를 bytes로 복원합니다.

            token 종류:
                - str:
                    <pad>, <unk>, <bos>, <eos> 같은 특수 토큰
                - bytes:
                    기본 byte token
                - tuple:
                    BPE merge token, 예: (69, 70)
            """

            token = self.id_to_token[token_id]

            # 특수 토큰입니다. 보통 decode 결과에서는 제외합니다.
            if isinstance(token, str):
                if skip_special:
                    return b""
                return token.encode("utf-8")

            # 기본 byte token입니다.
            # 예: id 69 -> b"A"
            if isinstance(token, bytes):
                return token

            # BPE merge token입니다.
            # 예: id 260 -> (69, 70)
            # 이 경우 왼쪽 token과 오른쪽 token을 각각 bytes로 펼친 뒤 이어 붙입니다.
            if isinstance(token, tuple):
                left_id, right_id = token
                left_bytes = token_to_bytes(left_id)
                right_bytes = token_to_bytes(right_id)
                return left_bytes + right_bytes

            raise TypeError(f"지원하지 않는 token 타입입니다: {type(token)}")

        byte_chunks = []

        for token_id in ids:
            token_bytes = token_to_bytes(token_id)
            byte_chunks.append(token_bytes)

        text_bytes = b"".join(byte_chunks)

        return text_bytes.decode("utf-8", errors="replace")
