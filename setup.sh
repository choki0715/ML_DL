#!/bin/bash
#===============================================================================
# ML_DL 실습 환경 설치 스크립트
#
# 이 저장소(ML_DL)의 노트북 실습에 필요한 파이썬 환경만 구성합니다.
# - 가상환경: $HOME/ai-training-env
# - 머신러닝: scikit-learn (회귀/KNN/결정트리/SVM/AdaBoost/군집화/차원축소)
# - 딥러닝  : PyTorch + torchvision (MLP/CNN/LSTM/seq2seq), TensorFlow
# - 강화학습: Gymnasium (CartPole / FrozenLake / Taxi Q-learning)
# - NLP/임베딩: gensim, nltk, tiktoken (토큰화 / word2vec)
# - Jupyter + VS Code 인터프리터 자동 설정 + 한글 그래프 폰트
#
# 사용법:  bash setup.sh
#          (Ubuntu 22.04 / 24.04, Python 3.10+)
#===============================================================================

set -e

# 안정 버전 (호환성 검증 조합)
TORCH_VERSION="2.6.0"
TORCHVISION_VERSION="0.21.0"
CUDA_INDEX="https://download.pytorch.org/whl/cu124"
VENV_DIR="${VENV_DIR:-$HOME/ai-training-env}"

# 색상
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'
log_info()    { echo -e "${BLUE}[i]${NC} $*"; }
log_success() { echo -e "${GREEN}[✓]${NC} $*"; }
log_warning() { echo -e "${YELLOW}[!]${NC} $*"; }
log_error()   { echo -e "${RED}[✗]${NC} $*"; }

#-------------------------------------------------------------------------------
# 1. Python 확인
#-------------------------------------------------------------------------------
detect_python() {
    for cand in python3.11 python3.12 python3.10 python3; do
        if command -v "$cand" &>/dev/null && "$cand" -c "import venv" &>/dev/null; then
            PYTHON_BIN="$(command -v "$cand")"
            log_success "Python 발견: $PYTHON_BIN ($($PYTHON_BIN --version))"
            return 0
        fi
    done
    log_error "Python 3.10+ (venv 포함) 이 필요합니다: sudo apt install python3.11 python3.11-venv"
    exit 1
}

#-------------------------------------------------------------------------------
# 2. 시스템 패키지 (한글 폰트 / graphviz / venv) — sudo 필요
#-------------------------------------------------------------------------------
install_system_deps() {
    log_info "시스템 패키지 설치 중 (한글 폰트 / graphviz)..."
    if command -v sudo &>/dev/null; then
        sudo apt-get update -qq || log_warning "apt update 실패 (계속 진행)"
        sudo apt-get install -y python3-venv fonts-nanum graphviz \
            || log_warning "일부 시스템 패키지 설치 실패 (계속 진행)"
        # 한글 폰트 캐시 갱신
        fc-cache -f &>/dev/null || true
    else
        log_warning "sudo 없음 — 한글 폰트/graphviz 는 수동 설치 필요"
    fi
}

#-------------------------------------------------------------------------------
# 3. 가상환경 생성
#-------------------------------------------------------------------------------
create_venv() {
    if [ -d "$VENV_DIR" ]; then
        log_warning "기존 가상환경 발견: $VENV_DIR (재사용)"
    else
        log_info "가상환경 생성: $VENV_DIR"
        "$PYTHON_BIN" -m venv "$VENV_DIR"
    fi
    # shellcheck disable=SC1091
    source "$VENV_DIR/bin/activate"
    pip install --upgrade pip setuptools wheel
}

#-------------------------------------------------------------------------------
# 4. 파이썬 패키지 설치 (이 폴더 노트북에 필요한 것만)
#-------------------------------------------------------------------------------
install_packages() {
    # GPU 유무 판정
    if command -v nvidia-smi &>/dev/null && nvidia-smi &>/dev/null; then
        HAS_GPU=true; log_info "NVIDIA GPU 감지됨 → CUDA 휠 설치"
    else
        HAS_GPU=false; log_info "GPU 미감지 → CPU 휠 설치"
    fi

    # PyTorch (+ torchvision)
    log_info "PyTorch ${TORCH_VERSION} 설치 중..."
    if [ "$HAS_GPU" = true ]; then
        pip install "torch==${TORCH_VERSION}" "torchvision==${TORCHVISION_VERSION}" \
            --index-url "${CUDA_INDEX}"
    else
        pip install "torch==${TORCH_VERSION}" "torchvision==${TORCHVISION_VERSION}" \
            --index-url https://download.pytorch.org/whl/cpu
    fi

    # 데이터 분석 / 시각화 (numpy<2 : TensorFlow 2.17 호환)
    log_info "데이터 분석 / 시각화 패키지 설치 중..."
    pip install "numpy<2" "pandas>=2.2" "scipy>=1.13" matplotlib seaborn tqdm

    # 머신러닝
    log_info "scikit-learn / 시각화 보조 설치 중..."
    pip install scikit-learn umap-learn graphviz

    # 딥러닝 (TensorFlow CPU 휠 — torch 의 CUDA 라이브러리와 충돌 방지)
    log_info "TensorFlow 설치 중..."
    pip install "tensorflow==2.17.0"

    # 강화학습
    log_info "Gymnasium (강화학습) 설치 중..."
    pip install "gymnasium[classic-control]" pygame

    # NLP / 임베딩
    log_info "NLP / 임베딩 패키지 설치 중..."
    pip install gensim nltk tiktoken
    python - <<'PY'
import nltk
for pkg in ['punkt', 'punkt_tab', 'stopwords']:
    try: nltk.download(pkg, quiet=True)
    except Exception: pass
PY

    # Jupyter (ipykernel 은 VS Code 호환 검증된 6.29.5 고정)
    log_info "Jupyter 환경 설치 중..."
    pip install jupyter jupyterlab ipywidgets "ipykernel==6.29.5"
    python -m ipykernel install --user --name=ai-training --display-name="ai-training-env"
}

#-------------------------------------------------------------------------------
# 5. VS Code 인터프리터 자동 지정
#   클론한 폴더에 .vscode/settings.json 생성 → 폴더 열면 자동으로 이 환경 선택
#   (torch 등 import 가 바로 인식되어 빨간 표시/커널 수동선택 불필요)
#-------------------------------------------------------------------------------
setup_vscode_settings() {
    local SCRIPT_DIR VENV_PY
    SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
    VENV_PY="$VENV_DIR/bin/python"
    [ -x "$VENV_PY" ] || { log_warning "venv 파이썬 없음 — VS Code 설정 건너뜀"; return 0; }

    mkdir -p "$SCRIPT_DIR/.vscode"
    cat > "$SCRIPT_DIR/.vscode/settings.json" <<EOF
{
    "python.defaultInterpreterPath": "$VENV_PY",
    "python.terminal.activateEnvironment": true,
    "jupyter.notebookFileRoot": "\${workspaceFolder}"
}
EOF
    log_success "VS Code 설정 생성: .vscode/settings.json (인터프리터: $VENV_PY)"
}

#-------------------------------------------------------------------------------
# 6. 설치 확인
#-------------------------------------------------------------------------------
verify() {
    log_info "설치된 주요 패키지 버전:"
    python - <<'PY'
def show(name):
    try:
        m = __import__(name); print(f'  {name:14s}: {getattr(m, "__version__", "?")}')
    except Exception as e:
        print(f'  {name:14s}: (오류 {e.__class__.__name__})')
import torch
print(f'  torch         : {torch.__version__}  (CUDA: {torch.cuda.is_available()})')
for p in ['torchvision','numpy','pandas','sklearn','tensorflow',
          'gymnasium','gensim','nltk','matplotlib']:
    show(p)
PY
}

#-------------------------------------------------------------------------------
# 메인
#-------------------------------------------------------------------------------
main() {
    echo ""
    log_info "===== ML_DL 실습 환경 설치 시작 ====="
    detect_python
    install_system_deps
    create_venv
    install_packages
    setup_vscode_settings
    verify
    echo ""
    log_success "설치 완료!"
    echo ""
    echo "  • 가상환경 활성화 : source $VENV_DIR/bin/activate"
    echo "  • Jupyter 실행     : jupyter lab"
    echo "  • VS Code          : 이 폴더를 열면 ai-training-env 가 자동 선택됩니다"
    echo ""
}

main "$@"
