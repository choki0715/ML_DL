#!/bin/bash

#===============================================================================
# ML_DL 실습 환경 설정 스크립트 (RTX 4060 / 8GB VRAM 타겟)
# Ubuntu 22.04 / 24.04 LTS 지원
#
# 포함 내용:
# 1. VSCode + Claude Code CLI
# 2. PyTorch (CUDA 12.4) + TensorFlow (CUDA 12)
# 3. Classical ML (scikit-learn, XGBoost, statsmodels)
# 4. CV (OpenCV, scikit-image, Ultralytics YOLO, TorchCAM, U-Net)
# 5. NLP / LLM (HuggingFace transformers/PEFT/TRL/bitsandbytes,
#    gensim, KoNLPy, sentence-transformers, GPTQ)
# 6. RAG (LangChain + FAISS) / Streamlit
# 7. RL (Gymnasium) / Audio (openai-whisper + ffmpeg)
#
# 변경 이력:
# v2.0 - Ubuntu 24.04 호환, 안정 버전 고정, Python 기본값 복구
# v2.1 - 노트북 다운로드 방식 변경
# v3.0 - Unsloth 제거, 패키지 버전 호환성 수정, RAG/KoNLPy 추가
#        transformers==4.46.0 + trl==0.12.0 + peft==0.14.0 호환 조합
#        멀티 GPU 충돌 방지 (CUDA_VISIBLE_DEVICES)
#        .env API 키 자동 로드
# v3.1 - 필요한 실습 파일만 다운로드, nltk 데이터 포함
# v3.2 - rouge-score, bert-score, matplotlib, llama-cpp-python 추가
# v4.0 - RTX 4060(Ada Lovelace, sm_89, 8GB VRAM) 환경 최적화
#        ML_DL 로컬 저장소 노트북 전 범위 지원
#        scikit-learn/xgboost/statsmodels, OpenCV(contrib), scikit-image,
#        TensorFlow(cu12), ultralytics, torchcam, optimum/auto-gptq,
#        openai-whisper(+ffmpeg), gymnasium[classic-control] 추가
#        VRAM 8GB 가이드 환경변수(PYTORCH_CUDA_ALLOC_CONF) 추가
#        프로젝트 셋업: 외부 다운로드 대신 로컬 ML_DL 폴더 사용
#===============================================================================

set -e

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 로깅 함수
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_banner() {
    echo ""
    echo "╔════════════════════════════════════════════════════════════════════╗"
    echo "║       ML_DL 실습 환경 설정 스크립트 (RTX 4060 타겟) v4.0          ║"
    echo "║                                                                    ║"
    echo "║  • PyTorch(cu124) + TensorFlow(cu12) + bitsandbytes(sm_89)         ║"
    echo "║  • Classical ML (sklearn / XGBoost / statsmodels)                  ║"
    echo "║  • CV (OpenCV / scikit-image / YOLO / TorchCAM)                    ║"
    echo "║  • NLP/LLM (HF / PEFT / TRL / GPTQ / KoNLPy)                       ║"
    echo "║  • RAG / Streamlit / Whisper / Gymnasium                           ║"
    echo "╚════════════════════════════════════════════════════════════════════╝"
    echo ""
}

#===============================================================================
# 안정 버전 정의 (호환성 테스트 완료 조합)
#===============================================================================
TORCH_VERSION="2.6.0"
TORCHVISION_VERSION="0.21.0"
CUDA_INDEX="https://download.pytorch.org/whl/cu124"
PYTHON_VERSION="3.11"

# HuggingFace 호환 조합 (테스트 완료)
TRANSFORMERS_VERSION="4.46.0"
TRL_VERSION="0.12.0"
PEFT_VERSION="0.14.0"
BITSANDBYTES_VERSION="0.45.0"

# venv 위치 / 사용 Python (detect_python에서 채워짐)
VENV_DIR="${VENV_DIR:-$HOME/ai-training-env}"
PYTHON_BIN=""

#-------------------------------------------------------------------------------
# 사용 가능한 Python 인터프리터 탐지 (sudo 불필요)
#  - 우선순위: python3.11 > 3.12 > 3.10 > 3 (단, 3.10 이상)
#-------------------------------------------------------------------------------
detect_python() {
    for cand in python3.11 python3.12 python3.10 python3; do
        if command -v "$cand" &> /dev/null; then
            ver=$("$cand" -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null || echo "")
            major=${ver%.*}; minor=${ver#*.}
            if [ "$major" = "3" ] && [ -n "$minor" ] && [ "$minor" -ge 10 ]; then
                if "$cand" -c "import venv" &> /dev/null; then
                    PYTHON_BIN="$cand"
                    log_info "사용할 Python: $PYTHON_BIN ($ver)"
                    return 0
                fi
            fi
        fi
    done
    log_error "Python 3.10+ (with venv) 가 필요합니다. 'sudo apt install python3.11 python3.11-venv' 또는 --llm-only 모드를 사용하세요."
    exit 1
}

#-------------------------------------------------------------------------------
# 시스템 사전 요구사항 확인
#-------------------------------------------------------------------------------
check_requirements() {
    log_info "시스템 요구사항 확인 중..."
    
    if [ -f /etc/os-release ]; then
        . /etc/os-release
        if [[ "$ID" != "ubuntu" ]]; then
            log_error "이 스크립트는 Ubuntu에서만 지원됩니다."
            exit 1
        fi
        log_info "OS: $PRETTY_NAME"
        
        if [[ "$VERSION_ID" != "24.04" && "$VERSION_ID" != "22.04" ]]; then
            log_warning "이 스크립트는 Ubuntu 22.04/24.04에서 테스트되었습니다. 현재: $VERSION_ID"
        fi
    fi
    
    TOTAL_MEM=$(free -g | awk '/^Mem:/{print $2}')
    log_info "시스템 메모리: ${TOTAL_MEM}GB"
    if [ "$TOTAL_MEM" -lt 16 ]; then
        log_warning "최소 16GB RAM을 권장합니다. 현재: ${TOTAL_MEM}GB"
    fi
    
    FREE_SPACE=$(df -BG / | awk 'NR==2 {print $4}' | sed 's/G//')
    log_info "사용 가능한 디스크 공간: ${FREE_SPACE}GB"
    if [ "$FREE_SPACE" -lt 50 ]; then
        log_warning "최소 50GB 여유 공간을 권장합니다. 현재: ${FREE_SPACE}GB"
    fi
    
    if command -v nvidia-smi &> /dev/null; then
        log_success "NVIDIA GPU 감지됨"
        nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
        HAS_GPU=true

        GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
        GPU_VRAM_MIB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
        GPU_VRAM_GB=$((GPU_VRAM_MIB / 1024))

        if echo "$GPU_NAME" | grep -qi "4060"; then
            log_success "RTX 4060 (Ada Lovelace, sm_89) 감지 → 권장 환경과 일치합니다."
        fi
        if [ "$GPU_VRAM_GB" -le 9 ]; then
            log_warning "VRAM ${GPU_VRAM_GB}GB 환경입니다. LLM은 4-bit(QLoRA) 또는 ≤3B 모델을 권장합니다."
            log_warning "  - batch_size, gradient_accumulation_steps, max_seq_length를 보수적으로 설정하세요."
            log_warning "  - PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True 가 자동 설정됩니다."
        fi
    else
        log_warning "NVIDIA GPU가 감지되지 않음. CPU 모드로 설치됩니다."
        HAS_GPU=false
    fi
}

#-------------------------------------------------------------------------------
# 시스템 패키지 업데이트
#-------------------------------------------------------------------------------
update_system() {
    log_info "시스템 패키지 업데이트 중..."
    sudo apt-get update
    sudo apt-get upgrade -y
    sudo apt-get install -y \
        curl wget git build-essential cmake pkg-config \
        software-properties-common apt-transport-https \
        ca-certificates gnupg lsb-release unzip jq \
        default-jdk \
        ffmpeg libsndfile1 \
        libgl1 libglib2.0-0 \
        libopenblas-dev liblapack-dev gfortran \
        swig
    log_success "시스템 업데이트 완료"
}

#-------------------------------------------------------------------------------
# VSCode 설치
#-------------------------------------------------------------------------------
install_vscode() {
    log_info "Visual Studio Code 설치 중..."
    
    if command -v code &> /dev/null; then
        log_success "VSCode가 이미 설치되어 있습니다."
        return
    fi
    
    wget -qO- https://packages.microsoft.com/keys/microsoft.asc | gpg --dearmor > packages.microsoft.gpg
    sudo install -D -o root -g root -m 644 packages.microsoft.gpg /etc/apt/keyrings/packages.microsoft.gpg
    rm packages.microsoft.gpg
    
    echo "deb [arch=amd64,arm64,armhf signed-by=/etc/apt/keyrings/packages.microsoft.gpg] https://packages.microsoft.com/repos/code stable main" | \
        sudo tee /etc/apt/sources.list.d/vscode.list > /dev/null
    
    sudo apt-get update
    sudo apt-get install -y code
    
    log_success "VSCode 설치 완료"
}

#-------------------------------------------------------------------------------
# Node.js 설치
#-------------------------------------------------------------------------------
install_nodejs() {
    log_info "Node.js 설치 중..."
    
    if command -v node &> /dev/null; then
        log_success "Node.js가 이미 설치되어 있습니다: $(node --version)"
        return
    fi
    
    curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
    sudo apt-get install -y nodejs
    
    log_success "Node.js $(node --version) 설치 완료"
}

#-------------------------------------------------------------------------------
# Claude Code CLI 설치
#-------------------------------------------------------------------------------
install_claude_code() {
    log_info "Claude Code CLI 설치 중..."
    
    if command -v claude &> /dev/null; then
        log_success "Claude Code가 이미 설치되어 있습니다."
        return
    fi
    
    sudo npm install -g @anthropic-ai/claude-code
    log_success "Claude Code CLI 설치 완료"
}

#-------------------------------------------------------------------------------
# Python 환경 설정
#-------------------------------------------------------------------------------
install_python() {
    log_info "Python 환경 설정 중..."
    
    sudo add-apt-repository -y ppa:deadsnakes/ppa
    sudo apt-get update
    sudo apt-get install -y \
        python3.11 python3.11-venv python3.11-dev python3-pip
    
    python3.11 -m pip install --upgrade pip --break-system-packages 2>/dev/null || \
    python3.11 -m pip install --upgrade pip
    
    log_success "Python $(python3.11 --version) 설치 완료"
}

#-------------------------------------------------------------------------------
# CUDA Toolkit 설치
#-------------------------------------------------------------------------------
install_cuda() {
    if [ "$HAS_GPU" = false ]; then
        log_warning "GPU가 없어 CUDA 설치를 건너뜁니다."
        return
    fi
    
    log_info "CUDA Toolkit 설치 중..."
    
    if command -v nvcc &> /dev/null; then
        log_success "CUDA가 이미 설치되어 있습니다."
        return
    fi
    
    . /etc/os-release
    if [[ "$VERSION_ID" == "24.04" ]]; then
        CUDA_REPO="ubuntu2404"
    else
        CUDA_REPO="ubuntu2204"
    fi
    
    wget "https://developer.download.nvidia.com/compute/cuda/repos/${CUDA_REPO}/x86_64/cuda-keyring_1.1-1_all.deb"
    sudo dpkg -i cuda-keyring_1.1-1_all.deb
    rm cuda-keyring_1.1-1_all.deb
    
    sudo apt-get update
    sudo apt-get install -y cuda-toolkit
    
    if ! grep -q "cuda" ~/.bashrc; then
        echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
        echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
    fi
    
    log_success "CUDA 설치 완료"
}

#-------------------------------------------------------------------------------
# LLM 환경 설정 (HuggingFace + PEFT)
#-------------------------------------------------------------------------------
setup_llm_environment() {
    log_info "LLM Fine-tuning 환경 설정 중..."

    [ -z "$PYTHON_BIN" ] && detect_python

    # GPU 플래그가 비어 있으면 nvidia-smi로 자동 판정 (--venv-only 단독 실행 대비)
    if [ -z "${HAS_GPU:-}" ]; then
        if command -v nvidia-smi &> /dev/null; then HAS_GPU=true; else HAS_GPU=false; fi
    fi

    if [ -d "$VENV_DIR" ]; then
        if [ "${REUSE_VENV:-0}" = "1" ]; then
            log_warning "기존 가상환경을 재사용합니다: $VENV_DIR"
        else
            log_warning "기존 가상환경을 발견했습니다. 백업 후 재생성합니다."
            mv "$VENV_DIR" "${VENV_DIR}.backup.$(date +%Y%m%d%H%M%S)"
        fi
    fi

    if [ ! -d "$VENV_DIR" ]; then
        log_info "가상환경 생성: $VENV_DIR (with $PYTHON_BIN)"
        "$PYTHON_BIN" -m venv "$VENV_DIR"
    fi
    # shellcheck disable=SC1091
    source "$VENV_DIR/bin/activate"

    pip install --upgrade pip setuptools wheel
    
    # PyTorch
    if [ "$HAS_GPU" = true ]; then
        log_info "GPU 환경으로 PyTorch ${TORCH_VERSION} 설치 중..."
        pip install \
            "torch==${TORCH_VERSION}" \
            "torchvision==${TORCHVISION_VERSION}" \
            "torchaudio==${TORCH_VERSION}" \
            --index-url "${CUDA_INDEX}"
    else
        log_info "CPU 환경으로 PyTorch ${TORCH_VERSION} 설치 중..."
        pip install \
            "torch==${TORCH_VERSION}" \
            "torchvision==${TORCHVISION_VERSION}" \
            "torchaudio==${TORCH_VERSION}" \
            --index-url https://download.pytorch.org/whl/cpu
    fi
    
    if [ "$HAS_GPU" = true ]; then
        python -c "import torch; assert torch.cuda.is_available(), 'GPU not detected!'; print(f'GPU: {torch.cuda.get_device_name(0)}')"
    fi
    
    # 데이터/시각화 공통
    # numpy<2 고정: TF 2.17 / pmdarima / 일부 강의용 노트북 호환
    log_info "데이터 분석 / 시각화 패키지 설치 중..."
    pip install \
        "numpy<2" "pandas>=2.2" "scipy>=1.13" \
        matplotlib seaborn plotly tqdm \
        jinja2 openpyxl

    # Classical ML / 시계열
    log_info "Classical ML / 시계열 패키지 설치 중 (sklearn/XGBoost/statsmodels)..."
    pip install \
        "scikit-learn>=1.5" xgboost lightgbm \
        statsmodels pmdarima

    # 컴퓨터 비전 (OpenCV / scikit-image / YOLO / TorchCAM)
    # opencv 4.11+ 는 numpy>=2 요구 → numpy<2 와 충돌. 4.10 으로 핀.
    log_info "컴퓨터 비전 패키지 설치 중..."
    pip install \
        "opencv-contrib-python==4.10.0.84" \
        "Pillow>=10" scikit-image imageio \
        torchcam ultralytics segmentation-models-pytorch albumentations

    # TensorFlow (CPU 휠 고정)
    # 사유: TF 2.17 의 [and-cuda] 는 CUDA 12.3 + cuDNN 8.9 의 nvidia-* 휠을 설치해
    #       PyTorch 2.6+cu124(CUDA 12.4 + cuDNN 9.1) 의 nvidia-* 를 덮어써 torch 가 깨짐.
    #       cuDNN 메이저 버전이 달라 GPU 공유가 불가능하므로, 학습용 환경에서는
    #       PyTorch 만 GPU 를 사용하고 TF 는 CPU 로 둔다 (TF 노트북 2개는 작은 규모).
    log_info "TensorFlow (CPU 휠) 설치 중..."
    pip install "tensorflow==2.17.0" "tensorflow-datasets>=4.9"

    # HuggingFace 핵심 (RTX 4060 / sm_89: bitsandbytes>=0.45 필수)
    log_info "HuggingFace / LLM 핵심 패키지 설치 중..."
    pip install \
        "transformers==${TRANSFORMERS_VERSION}" \
        "trl==${TRL_VERSION}" \
        "peft==${PEFT_VERSION}" \
        "bitsandbytes>=${BITSANDBYTES_VERSION}" \
        "datasets>=3.0.0" \
        "accelerate>=1.2.0" \
        "evaluate" \
        "sentencepiece" \
        "protobuf" \
        "safetensors" \
        "einops" \
        "tokenizers"

    # 양자화 (GPTQ / AWQ)
    log_info "양자화 패키지 설치 중 (optimum / auto-gptq)..."
    pip install "optimum>=1.23" \
        || log_warning "optimum 설치 실패"
    pip install auto-gptq \
        || log_warning "auto-gptq 빌드 실패 - GPTQ_example.ipynb 실행 시 'pip install auto-gptq' 재시도 필요"

    # NLP / 임베딩
    log_info "NLP / 임베딩 패키지 설치 중..."
    pip install \
        gensim nltk konlpy sentence-transformers fasttext-wheel

    # nltk 데이터 다운로드
    python -c "import nltk; [nltk.download(p, quiet=True) for p in ['punkt','punkt_tab','stopwords','averaged_perceptron_tagger','wordnet']]"

    # RAG
    log_info "RAG 패키지 설치 중..."
    pip install \
        langchain langchain-community langchain-openai \
        langchain-text-splitters faiss-cpu \
        pypdf python-dotenv beautifulsoup4 requests \
        chromadb tiktoken

    # 평가 메트릭 / GGUF / 오디오 / 강화학습
    log_info "평가 / 오디오(Whisper) / 강화학습 패키지 설치 중..."
    pip install rouge-score bert-score sacrebleu llama-cpp-python
    pip install openai-whisper soundfile librosa
    pip install "gymnasium[classic-control]" pygame

    # Jupyter
    log_info "Jupyter 환경 설치 중..."
    # ipykernel은 VS Code Jupyter 확장 호환성이 검증된 6.29.x로 고정
    # (7.x는 "notebook controller is DISPOSED" 오류를 유발하는 경우가 있음)
    pip install jupyter jupyterlab ipywidgets "ipykernel==6.29.5"
    python -m ipykernel install --user --name=ai-training --display-name="ai-training-env"

    # Streamlit
    log_info "Streamlit 설치 중..."
    pip install streamlit
    
    # 확인
    log_info "설치된 주요 패키지 버전:"
    python - <<'PYCHK'
def show(name, attr='__version__'):
    try:
        m = __import__(name)
        print(f'  {name:22s}: {getattr(m, attr, "?")}')
    except Exception as e:
        print(f'  {name:22s}: (미설치/오류 - {e.__class__.__name__})')

import torch
print(f'  PyTorch              : {torch.__version__}  (CUDA: {torch.cuda.is_available()})')
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        cap = torch.cuda.get_device_capability(i)
        print(f'    GPU {i}: {torch.cuda.get_device_name(i)}  sm_{cap[0]}{cap[1]}')

for pkg in ['transformers','trl','peft','bitsandbytes','accelerate','datasets',
            'tensorflow','sklearn','xgboost','statsmodels',
            'cv2','skimage','ultralytics','torchcam',
            'gensim','nltk','konlpy','sentence_transformers',
            'langchain','faiss','streamlit',
            'gymnasium','whisper']:
    show(pkg)
PYCHK
    
    deactivate
    log_success "LLM Fine-tuning 환경 설정 완료"
}

#-------------------------------------------------------------------------------
# 실습 프로젝트 설정 (로컬 ML_DL 저장소 사용)
#-------------------------------------------------------------------------------
setup_example_project() {
    log_info "예제 프로젝트 설정 중..."

    PROJECT_DIR="$HOME/ai-training-projects"
    mkdir -p "$PROJECT_DIR"

    # 스크립트가 위치한 디렉터리(= ML_DL 저장소 루트로 가정)
    SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

    if [ -f "$SCRIPT_DIR/setup.sh" ] && ls "$SCRIPT_DIR"/*.ipynb >/dev/null 2>&1; then
        log_info "로컬 ML_DL 저장소 감지: $SCRIPT_DIR"
        ln -sfn "$SCRIPT_DIR" "$PROJECT_DIR/ML_DL"
        log_success "심볼릭 링크 생성: $PROJECT_DIR/ML_DL → $SCRIPT_DIR"
    else
        log_warning "ML_DL 노트북을 찾지 못했습니다. git clone을 시도합니다."
        if [ ! -d "$PROJECT_DIR/ML_DL" ]; then
            git clone https://github.com/choki0715/ML_DL.git "$PROJECT_DIR/ML_DL" \
                || log_warning "git clone 실패 - 수동으로 ML_DL 저장소를 배치하세요."
        fi
    fi

    # 작업/캐시 디렉터리
    mkdir -p "$PROJECT_DIR/ai-assistant"
    mkdir -p "$PROJECT_DIR/data"
    mkdir -p "$PROJECT_DIR/checkpoints"

    # .env 템플릿 (없을 때만)
    if [ ! -f "$PROJECT_DIR/.env" ]; then
        cat > "$PROJECT_DIR/.env" <<'ENVEOF'
# API 키는 사용 시점에만 입력하세요. 커밋 금지.
OPENAI_API_KEY=
ANTHROPIC_API_KEY=
HF_TOKEN=
ENVEOF
        chmod 600 "$PROJECT_DIR/.env"
        log_info ".env 템플릿 생성: $PROJECT_DIR/.env"
    fi

    log_success "예제 프로젝트 설정 완료: $PROJECT_DIR"
}

#-------------------------------------------------------------------------------
# VS Code 인터프리터 자동 지정
#   클론한 ML_DL 폴더에 .vscode/settings.json 을 생성해 가상환경 파이썬을
#   기본 인터프리터로 지정한다. 이러면 수강생이 폴더를 열자마자 torch 등
#   import 가 정상 인식되어 빨간 표시/커널 수동선택이 필요 없다.
#-------------------------------------------------------------------------------
setup_vscode_settings() {
    log_info "VS Code 인터프리터 설정 중..."

    local SCRIPT_DIR
    SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
    local VENV_PY="$VENV_DIR/bin/python"

    if [ ! -x "$VENV_PY" ]; then
        log_warning "가상환경 파이썬을 찾지 못해 VS Code 설정을 건너뜁니다: $VENV_PY"
        return 0
    fi

    mkdir -p "$SCRIPT_DIR/.vscode"
    cat > "$SCRIPT_DIR/.vscode/settings.json" <<EOF
{
    "python.defaultInterpreterPath": "$VENV_PY",
    "python.terminal.activateEnvironment": true,
    "jupyter.notebookFileRoot": "\${workspaceFolder}"
}
EOF
    log_success "VS Code 설정 생성: $SCRIPT_DIR/.vscode/settings.json (인터프리터: $VENV_PY)"
    log_info "VS Code에서 이 폴더를 열면 자동으로 ai-training-env 가 선택됩니다."
}

#-------------------------------------------------------------------------------
# 환경 변수 설정
#-------------------------------------------------------------------------------
setup_environment_variables() {
    log_info "환경 변수 설정 중..."
    
    if grep -q "AI Training Environment" ~/.bashrc 2>/dev/null; then
        log_warning "환경 변수가 이미 설정되어 있습니다. 건너뜁니다."
        return
    fi
    
    cat >> ~/.bashrc << 'EOF'

# ====== AI Training Environment v4.0 (RTX 4060) ======
export AI_TRAINING_HOME="$HOME/ai-training-projects"
export AI_VENV="$HOME/ai-training-env"

# 멀티 GPU 충돌 방지 (기본 GPU 0번만 사용)
export CUDA_VISIBLE_DEVICES=0

# RTX 4060 / 8GB VRAM 친화 설정
# - PyTorch 단편화 완화 (긴 컨텍스트 / 동적 shape 워크로드)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# - TensorFlow가 VRAM을 한꺼번에 잡지 않도록
export TF_FORCE_GPU_ALLOW_GROWTH=true
# - TF 로그 노이즈 감소
export TF_CPP_MIN_LOG_LEVEL=2
# - tokenizers fork 경고 억제
export TOKENIZERS_PARALLELISM=false

# .env에서 API 키 자동 로드 (값에 공백/특수문자 있어도 안전하게)
if [ -f "$HOME/ai-training-projects/.env" ]; then
    set -a
    . "$HOME/ai-training-projects/.env"
    set +a
fi

# HuggingFace 캐시
export HF_HOME="$HOME/.cache/huggingface"
export TRANSFORMERS_CACHE="$HOME/.cache/huggingface/transformers"

# wandb 비활성화
export WANDB_DISABLED=true

# 가상환경 활성화: source ~/ai-training-env/bin/activate
EOF

    log_success "환경 변수 설정 완료"
}

#-------------------------------------------------------------------------------
# 설치 완료 정보 출력
#-------------------------------------------------------------------------------
print_completion_info() {
    echo ""
    echo "╔════════════════════════════════════════════════════════════════════╗"
    echo "║                    ✅ 설치가 완료되었습니다!                       ║"
    echo "╚════════════════════════════════════════════════════════════════════╝"
    echo ""
    echo "📁 설치된 위치:"
    echo "   • 가상환경 : $HOME/ai-training-env"
    echo "   • 프로젝트 : $HOME/ai-training-projects"
    echo "   • ML_DL    : $HOME/ai-training-projects/ML_DL  (로컬 저장소 심볼릭 링크)"
    echo ""
    echo "🚀 빠른 시작:"
    echo "   1. source ~/.bashrc"
    echo "   2. source ~/ai-training-env/bin/activate     (가상환경 활성화)"
    echo "   3. jupyter lab --ip=0.0.0.0 --no-browser     (Jupyter Lab)"
    echo "   4. streamlit run app.py                      (Streamlit)"
    echo "   5. claude                                    (Claude Code)"
    echo ""
    echo "   해제: deactivate"
    echo ""
    echo "🎮 RTX 4060 / 8GB VRAM 운영 팁:"
    echo "   • LLM 학습: 4-bit(QLoRA) + per_device_batch=1, grad_accum 사용"
    echo "   • 추론   : torch_dtype=torch.float16 또는 bfloat16, 모델 ≤ 7B(4-bit) 권장"
    echo "   • TF/PT 동시 사용 금지(같은 GPU에서 OOM 위험) — 노트북 커널 분리"
    echo "   • PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True 자동 적용됨"
    echo ""
    echo "📓 노트북 카테고리 (전부 ML_DL 폴더):"
    echo "   • Classical ML : IRIS_KNN, Decision_Tree, SVM*, XGBoost_Example,"
    echo "                    dimensionality_reduction, SVD, Unsupervised*"
    echo "   • 시계열       : AR_ARMA_ARIMA, time_series_decompose, multivariate_time_series"
    echo "   • DL 기초      : MLP/CNN_MNIST_PyTorch, fashion_minist_*"
    echo "   • 생성모델     : Convolutional_GAN/VAE_Fashion_MNIST, TF_GAN_Celeb"
    echo "   • CV           : cv2_image, HOG, opencv_*, TorchCAM_Quickstart, yolo_basics, Unet"
    echo "   • NLP/LLM      : tokenization, word2vec_embedding, fasttext_embedding,"
    echo "                    seq2seq_example, GPTQ_example, lora_finetuning"
    echo "   • RL / Audio   : RL_Gym_*, openai_whisper"
    echo "   • Web          : app.py (Streamlit)"
    echo ""
    echo "📋 API 키 설정: ~/ai-training-projects/.env"
    echo "   OPENAI_API_KEY=sk-..."
    echo "   ANTHROPIC_API_KEY=sk-ant-..."
    echo "   HF_TOKEN=hf_..."
    echo ""
}

#-------------------------------------------------------------------------------
# 메인 실행
#-------------------------------------------------------------------------------
main() {
    print_banner
    check_requirements
    update_system
    install_vscode
    install_nodejs
    install_claude_code
    install_python
    install_cuda
    setup_llm_environment
    setup_example_project
    setup_vscode_settings
    setup_environment_variables
    print_completion_info
}

case "${1:-}" in
    --vscode-only)   update_system; install_vscode ;;
    --claude-only)   install_nodejs; install_claude_code ;;
    --llm-only)      check_requirements; install_python; install_cuda; setup_llm_environment; setup_vscode_settings ;;
    --venv-only)
        # sudo 없이 가상환경 + 모든 라이브러리만 설치
        # 전제: Python 3.10+ 와 NVIDIA 드라이버가 이미 설치되어 있음
        print_banner
        check_requirements
        detect_python
        setup_llm_environment
        setup_vscode_settings
        setup_environment_variables
        echo ""
        log_success "가상환경 준비 완료: $VENV_DIR"
        log_info "활성화: source $VENV_DIR/bin/activate"
        ;;
    --venv-reuse)
        # 기존 venv를 지우지 않고 누락 패키지만 추가 설치
        REUSE_VENV=1
        check_requirements
        detect_python
        setup_llm_environment
        setup_vscode_settings
        ;;
    --project-only)  setup_example_project ;;
    --help)
        echo "사용법: $0 [옵션]"
        echo "  (없음)            전체 설치 (sudo 필요)"
        echo "  --vscode-only     VSCode만 설치 (sudo 필요)"
        echo "  --claude-only     Claude Code만 설치 (sudo 필요)"
        echo "  --llm-only        Python/CUDA 설치 + 가상환경 + 라이브러리 (sudo 필요)"
        echo "  --venv-only       가상환경 + 라이브러리만 설치 (sudo 불필요)"
        echo "  --venv-reuse      기존 가상환경 유지하고 라이브러리만 재설치"
        echo "  --project-only    프로젝트 폴더/.env만 설정"
        echo "  --help            도움말 표시"
        echo ""
        echo "환경변수:"
        echo "  VENV_DIR=...      가상환경 경로 변경 (기본: \$HOME/ai-training-env)"
        ;;
    *) main ;;
esac