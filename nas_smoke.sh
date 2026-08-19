#!/usr/bin/env bash
# ==============================================================================
# NAS(SFTP/SSH)에서 평가용 프레임을 "선택적으로" 받아 slot_v3 스모크 테스트
#
# 전체를 받지 않고 ① 폴더 몇 개만 ② 폴더당 앞 N프레임(연속 구간)만 가져옵니다.
# slot_v3는 --window 24로 직전 프레임을 참조하므로 무작위 샘플이 아니라
# 반드시 '연속 구간'이어야 시간축 로직이 정상 동작합니다.
#
# 사용법:
#   ./nas_smoke.sh <user@nas> <NAS_원격경로> [폴더수] [폴더당_프레임수]
# 예:
#   ./nas_smoke.sh irteam@10.0.0.5 /volume1/frames/airtel 3 200
#
# 환경변수:
#   PORT=22          ssh 포트
#   LOCAL=...        로컬 저장 경로 (기본 ./nas_data/<원격폴더명>)
#   OUT=...          평가 결과 경로 (기본 results/nas_smoke)
#   LIST_ONLY=1      다운로드 없이 NAS 폴더 목록만 확인
#   NO_EVAL=1        다운로드만 하고 평가는 생략
# ==============================================================================
set -euo pipefail

HOST="${1:?사용법: ./nas_smoke.sh <user@nas> <원격경로> [폴더수] [폴더당프레임수]}"
RPATH="${2:?원격 경로를 지정하세요 (예: /volume1/frames/airtel)}"
NFOLD="${3:-3}"
NFRAME="${4:-200}"
PORT="${PORT:-22}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$HERE"
LOCAL="${LOCAL:-$HERE/nas_data/$(basename "$RPATH")}"
OUT="${OUT:-results/nas_smoke}"
# ControlMaster: 비밀번호 인증이어도 최초 1회만 입력하면 이후 전송은 재사용
CM="/tmp/nas_cm_$$"
SSH="ssh -p $PORT -o StrictHostKeyChecking=accept-new -o ControlMaster=auto -o ControlPath=$CM -o ControlPersist=600"
cleanup() { ssh -O exit -o ControlPath="$CM" "$HOST" 2>/dev/null || true; }
trap cleanup EXIT

echo "[nas] host=$HOST  remote=$RPATH  폴더=$NFOLD개  폴더당=$NFRAME프레임"

# ── 1) NAS의 하위 폴더 목록 (프레임이 폴더별로 나뉘어 있다고 가정) ──────────────
echo "[nas] 폴더 목록 조회..."
mapfile -t FOLDERS < <($SSH "$HOST" "find '$RPATH' -mindepth 1 -maxdepth 1 -type d | sort")

if [ "${#FOLDERS[@]}" -eq 0 ]; then
  echo "[nas] 하위 폴더 없음 → '$RPATH' 자체를 하나의 프레임 폴더로 취급"
  FOLDERS=("$RPATH")
fi

echo "[nas] 총 ${#FOLDERS[@]}개 폴더 발견. 앞 $NFOLD개 사용:"
PICK=("${FOLDERS[@]:0:$NFOLD}")
for f in "${PICK[@]}"; do echo "        - $f"; done

if [ "${LIST_ONLY:-0}" = "1" ]; then
  echo "[nas] LIST_ONLY=1 → 여기서 종료. 원하는 폴더를 골라 다시 실행하세요."
  exit 0
fi

# ── 2) 폴더별로 '앞 N개 연속 프레임'만 골라서 전송 ────────────────────────────
mkdir -p "$LOCAL"
for f in "${PICK[@]}"; do
  name="$(basename "$f")"
  echo "[nas] $name ← 파일명 정렬 후 앞 $NFRAME개 전송"

  # 파일명 정렬 = 프레임 순서. 앞에서 연속으로 끊어야 시간축이 유지됨.
  $SSH "$HOST" "find '$f' -maxdepth 1 -type f \
      \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' -o -iname '*.bmp' \) \
      -printf '%f\n' | sort | head -n $NFRAME" > "/tmp/nas_list_$$.txt"

  cnt=$(wc -l < "/tmp/nas_list_$$.txt")
  if [ "$cnt" -eq 0 ]; then echo "[nas]   (이미지 없음, 건너뜀)"; continue; fi

  mkdir -p "$LOCAL/$name"
  rsync -a --info=progress2 -e "$SSH" \
        --files-from="/tmp/nas_list_$$.txt" \
        "$HOST:$f/" "$LOCAL/$name/"
  rm -f "/tmp/nas_list_$$.txt"
  echo "[nas]   → $LOCAL/$name ($cnt장)"
done

echo "[nas] 다운로드 완료: $(du -sh "$LOCAL" | cut -f1)"

# ── 3) 평가 실행 ─────────────────────────────────────────────────────────────
if [ "${NO_EVAL:-0}" = "1" ]; then
  echo "[nas] NO_EVAL=1 → 평가 생략. 수동 실행:"
  echo "      PYTHON=\$(which python) ./run_slot_v3_ft.sh '$LOCAL' '$OUT'"
  exit 0
fi

echo "[nas] slot_v3 평가 시작..."
PYTHON="${PYTHON:-$(command -v python)}" ./run_slot_v3_ft.sh "$LOCAL" "$OUT"
