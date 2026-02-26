#!/bin/bash

##############################################################################
# AI Video Upscaler - Automated Installation Script
# For Ubuntu 24.04 LTS with NVIDIA GPU
# Updated for latest stable drivers and packages (February 2026)
##############################################################################

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_header() {
    echo -e "\n${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}\n"
}

print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

check_ubuntu() {
    if [ -f /etc/os-release ]; then
        . /etc/os-release
        if [[ "$ID" != "ubuntu" ]]; then
            print_error "This script is designed for Ubuntu. Detected: $ID"
            exit 1
        fi
        print_success "Ubuntu detected: $VERSION"

        # Recommend Ubuntu 24.04 LTS
        VERSION_ID_NUM="${VERSION_ID//./}"
        if (( VERSION_ID_NUM < 2404 )); then
            print_warning "Ubuntu 24.04 LTS is recommended for latest drivers and packages"
            print_warning "You are running Ubuntu $VERSION_ID"
        fi
    else
        print_error "Cannot determine OS version"
        exit 1
    fi
}

check_nvidia() {
    if ! lspci | grep -i nvidia > /dev/null; then
        print_error "No NVIDIA GPU detected"
        exit 1
    fi
    print_success "NVIDIA GPU detected"
}

##############################################################################
# Installation Steps
##############################################################################

install_nvidia_driver() {
    print_header "Installing NVIDIA Driver"

    if command -v nvidia-smi &> /dev/null; then
        print_warning "NVIDIA driver already installed"
        nvidia-smi --query-gpu=driver_version --format=csv,noheader
        return 0
    fi

    print_info "Detecting recommended NVIDIA driver for your GPU..."

    # Install ubuntu-drivers-common if not present
    sudo apt install -y ubuntu-drivers-common

    # Show recommended driver
    RECOMMENDED_DRIVER=$(ubuntu-drivers devices 2>/dev/null | grep recommended | awk '{print $3}')

    if [ -z "$RECOMMENDED_DRIVER" ]; then
        print_info "Using ubuntu-drivers to auto-install latest stable driver..."
        sudo ubuntu-drivers install
    else
        print_info "Installing recommended driver: $RECOMMENDED_DRIVER"
        sudo apt install -y "$RECOMMENDED_DRIVER"
    fi

    print_warning "System needs to reboot to load NVIDIA driver"
    print_warning "After reboot, run this script again with: ./install.sh --resume"

    read -p "Reboot now? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        sudo reboot
    else
        print_info "Please reboot manually and run: ./install.sh --resume"
        exit 0
    fi
}

install_cuda() {
    print_header "Installing CUDA Toolkit"

    if [ -f /usr/local/cuda/bin/nvcc ]; then
        print_warning "CUDA already installed"
        /usr/local/cuda/bin/nvcc --version | grep "release"
        return 0
    fi

    print_info "Downloading CUDA keyring for Ubuntu 24.04..."
    wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
    sudo dpkg -i cuda-keyring_1.1-1_all.deb
    rm cuda-keyring_1.1-1_all.deb

    print_info "Installing CUDA toolkit 12.6 (this may take 10-15 minutes)..."
    sudo apt update
    sudo apt install -y cuda-toolkit-12-6

    print_info "Adding CUDA to PATH..."
    if ! grep -q "cuda/bin" ~/.bashrc; then
        echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
        echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
    fi

    export PATH=/usr/local/cuda/bin:$PATH
    export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

    print_success "CUDA installed successfully"
}

install_dependencies() {
    print_header "Installing System Dependencies"

    print_info "Updating package lists..."
    sudo apt update

    print_info "Installing Python 3.12, FFmpeg 6.x, and build tools..."
    sudo apt install -y \
        python3 \
        python3-pip \
        python3-venv \
        python3-dev \
        ffmpeg \
        build-essential \
        cmake \
        git \
        wget \
        curl \
        mediainfo \
        bc \
        pkg-config

    print_success "Dependencies installed"
}

setup_ai_upscale() {
    print_header "Setting up AI Upscale Environment"

    INSTALL_DIR="$HOME/ai-upscale"
    print_info "Creating directory: $INSTALL_DIR"
    mkdir -p "$INSTALL_DIR"
    cd "$INSTALL_DIR"

    print_info "Creating Python virtual environment..."
    if [ ! -d "venv" ]; then
        python3 -m venv venv
    fi

    source venv/bin/activate

    print_info "Upgrading pip to latest version..."
    pip install --upgrade pip setuptools wheel

    print_info "Installing PyTorch with CUDA 12.4 support (latest stable)..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

    print_info "Installing spandrel (universal model loader) and dependencies..."
    pip install spandrel spandrel-extra-arches
    pip install opencv-python
    pip install tqdm
    pip install numpy

    print_success "Python environment configured"

    # Verify CUDA availability
    print_info "Verifying CUDA availability..."
    python3 << EOF
import torch
if torch.cuda.is_available():
    print(f"✓ CUDA is available")
    print(f"✓ CUDA version: {torch.version.cuda}")
    print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
else:
    print("✗ CUDA is NOT available - AI upscaling will be very slow!")
    exit(1)
EOF

    if [ $? -ne 0 ]; then
        print_error "CUDA verification failed"
        exit 1
    fi

    print_info "Verifying spandrel installation..."
    python3 -c "from spandrel import ModelLoader; print('✓ spandrel OK')"
}

download_model() {
    local filename="$1"
    local url="$2"
    local description="$3"
    local max_retries=3
    local retry_count=0

    if [ -f "$filename" ]; then
        print_warning "$filename already exists, skipping"
        return 0
    fi

    print_info "Downloading $description..."

    while [ $retry_count -lt $max_retries ]; do
        if wget --show-progress -O "$filename.tmp" "$url" 2>&1; then
            mv "$filename.tmp" "$filename"
            print_success "$filename downloaded successfully"
            return 0
        else
            retry_count=$((retry_count + 1))
            rm -f "$filename.tmp"
            if [ $retry_count -lt $max_retries ]; then
                print_warning "Download failed, retrying ($retry_count/$max_retries)..."
                sleep 2
            fi
        fi
    done

    print_error "Failed to download $filename after $max_retries attempts"
    print_error "URL: $url"
    print_warning "You can manually download this file later"
    return 1
}

download_models() {
    print_header "Downloading AI Models"

    INSTALL_DIR="$HOME/ai-upscale"
    MODEL_DIR="$INSTALL_DIR/models"

    mkdir -p "$MODEL_DIR"
    cd "$MODEL_DIR"

    print_info "Downloading models..."
    print_info "This may take several minutes depending on your internet speed..."
    echo ""

    # Temporarily disable exit on error for downloads
    set +e

    # RealESRGAN_x4plus — legacy fallback, confirmed direct download
    download_model \
        "RealESRGAN_x4plus.pth" \
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth" \
        "RealESRGAN_x4plus (legacy fallback)"

    # 4x-UltraSharp — direct download from HuggingFace
    download_model \
        "4x-UltraSharp.pth" \
        "https://huggingface.co/Kim2091/UltraSharp/resolve/main/4x-UltraSharp.pth" \
        "4x-UltraSharp"

    # Re-enable exit on error
    set -e

    echo ""
    print_warning "The recommended default model (4xNomos8kSC.pth) must be downloaded manually:"
    print_info "  1. Go to: https://openmodeldb.info"
    print_info "  2. Search for '4xNomos8kSC'"
    print_info "  3. Download 4xNomos8kSC.pth → place in $MODEL_DIR"
    echo ""
    print_warning "Other optional models (also from openmodeldb.info):"
    print_info "  - 4xLSDIR.pth       (model key: lsdir)"
    print_info "  - HAT-L_SRx4_ImageNet-pretrain.pth  (model key: hat)"
    echo ""

    if [ ! -f "RealESRGAN_x4plus.pth" ] && [ ! -f "4xNomos8kSC.pth" ]; then
        print_error "No models found in $MODEL_DIR"
        print_error "Download at least one model before upscaling"
    else
        print_success "Model downloads complete!"
        print_info "Downloaded models:"
        ls -lh *.pth 2>/dev/null | awk '{print "  - " $9 " (" $5 ")"}'
    fi

    cd "$INSTALL_DIR"
}

copy_scripts() {
    print_header "Installing Upscale Script"

    INSTALL_DIR="$HOME/ai-upscale"
    SCRIPT_FOUND=false

    # Try multiple locations for upscale_video.sh
    for path in "./upscale_video.sh" "$(dirname "$0")/upscale_video.sh" "$HOME/Downloads/upscale_video.sh"; do
        if [ -f "$path" ]; then
            print_info "Found upscale_video.sh at: $path"
            print_info "Copying to $INSTALL_DIR..."
            cp "$path" "$INSTALL_DIR/"
            chmod +x "$INSTALL_DIR/upscale_video.sh"
            print_success "Upscale script installed"
            SCRIPT_FOUND=true
            break
        fi
    done

    if [ "$SCRIPT_FOUND" = false ]; then
        print_warning "upscale_video.sh not found in common locations"
        print_info "Please copy upscale_video.sh to $INSTALL_DIR manually:"
        echo "  cp upscale_video.sh $INSTALL_DIR/"
        echo "  chmod +x $INSTALL_DIR/upscale_video.sh"
    fi
}

optimize_gpu() {
    print_header "GPU Optimization (Optional)"

    print_info "Setting GPU to performance mode..."
    if sudo nvidia-smi -pm 1 2>/dev/null; then
        print_success "Persistence mode enabled"
    fi

    print_info "Setting power limit to maximum..."
    if sudo nvidia-smi -pl 165 2>/dev/null; then
        print_success "Power limit set to 165W"
    fi
}

create_test_script() {
    print_header "Creating Test Script"

    INSTALL_DIR="$HOME/ai-upscale"

    cat > "$INSTALL_DIR/test.sh" << 'EOF'
#!/bin/bash
echo "Testing AI Upscale Installation..."
echo ""

# Test 1: NVIDIA driver
echo -n "1. NVIDIA driver: "
if nvidia-smi &> /dev/null; then
    DRIVER_VER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null)
    echo "✓ OK (driver $DRIVER_VER)"
else
    echo "✗ FAILED — run: sudo ubuntu-drivers autoinstall && sudo reboot"
fi

# Test 2: CUDA toolkit
echo -n "2. CUDA toolkit: "
if command -v nvcc &> /dev/null; then
    CUDA_VER=$(nvcc --version 2>/dev/null | grep "release" | awk '{print $6}' | tr -d ,)
    echo "✓ OK ($CUDA_VER)"
elif [ -f /usr/local/cuda/bin/nvcc ]; then
    echo "✓ OK (at /usr/local/cuda/bin/nvcc — add to PATH if needed)"
else
    echo "✗ FAILED — CUDA not found"
fi

# Test 3: FFmpeg
echo -n "3. FFmpeg: "
if ffmpeg -version &> /dev/null; then
    FFMPEG_VER=$(ffmpeg -version 2>/dev/null | head -1 | awk '{print $3}')
    echo "✓ OK ($FFMPEG_VER)"
else
    echo "✗ FAILED — run: sudo apt install ffmpeg"
fi

# Test 4: Python environment
echo -n "4. Python venv: "
if [ -f ~/ai-upscale/venv/bin/python3 ]; then
    PY_VER=$(~/ai-upscale/venv/bin/python3 --version 2>/dev/null)
    echo "✓ OK ($PY_VER)"
else
    echo "✗ FAILED — run install.sh"
fi

# Test 5: PyTorch CUDA
echo -n "5. PyTorch CUDA: "
if [ -f ~/ai-upscale/venv/bin/python3 ]; then
    if ~/ai-upscale/venv/bin/python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
        GPU_NAME=$(~/ai-upscale/venv/bin/python3 -c "import torch; print(torch.cuda.get_device_name(0))" 2>/dev/null)
        TORCH_VER=$(~/ai-upscale/venv/bin/python3 -c "import torch; print(torch.__version__)" 2>/dev/null)
        echo "✓ OK (PyTorch $TORCH_VER — $GPU_NAME)"
    else
        echo "✗ FAILED — CUDA not available in PyTorch"
        echo "   Fix: source ~/ai-upscale/venv/bin/activate"
        echo "        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124"
    fi
else
    echo "✗ FAILED (venv not found)"
fi

# Test 6: spandrel
echo -n "6. spandrel: "
if [ -f ~/ai-upscale/venv/bin/python3 ]; then
    if ~/ai-upscale/venv/bin/python3 -c "from spandrel import ModelLoader" 2>/dev/null; then
        echo "✓ OK"
    else
        echo "✗ FAILED — run: source ~/ai-upscale/venv/bin/activate && pip install spandrel spandrel-extra-arches"
    fi
else
    echo "✗ FAILED (venv not found)"
fi

# Test 7: AI models
echo -n "7. AI models: "
MODEL_COUNT=$(ls ~/ai-upscale/models/*.pth 2>/dev/null | wc -l)
if [ "$MODEL_COUNT" -gt 0 ]; then
    echo "✓ OK ($MODEL_COUNT model(s) found)"
    ls ~/ai-upscale/models/*.pth 2>/dev/null | while read f; do
        SIZE=$(du -h "$f" | cut -f1)
        echo "     $(basename "$f") ($SIZE)"
    done
else
    echo "✗ No models found in ~/ai-upscale/models/"
    echo "   Download 4xNomos8kSC.pth from openmodeldb.info and place it there"
fi

# Test 8: Upscale script
echo -n "8. Upscale script: "
THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -x "$THIS_DIR/upscale_video.sh" ]; then
    echo "✓ OK"
elif [ -x ~/ai-upscale/upscale_video.sh ]; then
    echo "✓ OK"
else
    echo "✗ FAILED — copy upscale_video.sh to ~/ai-upscale/ and chmod +x it"
fi

echo ""
echo "Test complete."
EOF

    chmod +x "$INSTALL_DIR/test.sh"
    print_success "Test script created: $INSTALL_DIR/test.sh"
}

show_completion() {
    print_header "Installation Complete!"

    echo -e "${GREEN}Installation complete!${NC}\n"
    echo -e "Directory: ${BLUE}$HOME/ai-upscale${NC}\n"
    echo -e "${YELLOW}Next step — download your model:${NC}"
    echo "  Go to https://openmodeldb.info and download 4xNomos8kSC.pth"
    echo -e "  Place it in $HOME/ai-upscale/models/\n"
    echo -e "${YELLOW}Test the installation:${NC}"
    echo "  cd ~/ai-upscale"
    echo -e "  ./test.sh\n"
    echo -e "${YELLOW}Basic usage:${NC}"
    echo "  ./upscale_video.sh -i input.mkv -r 1080p"
    echo -e "  ./upscale_video.sh --help\n"
    echo -e "${YELLOW}Documentation:${NC}"
    echo -e "  See QUICKSTART.md and README.md\n"
}

##############################################################################
# Main Installation Flow
##############################################################################

main() {
    print_header "AI Video Upscaler - Installation"

    # Check if running as resume after reboot
    RESUME=false
    if [[ "$1" == "--resume" ]]; then
        RESUME=true
        print_info "Resuming installation after reboot..."
    fi

    # Pre-flight checks
    check_ubuntu
    check_nvidia

    # Update system
    print_header "Updating System"
    sudo apt update
    sudo apt upgrade -y

    # Install NVIDIA driver (may require reboot)
    if [ "$RESUME" = false ]; then
        install_nvidia_driver
    fi

    # Verify NVIDIA driver is loaded
    if ! nvidia-smi &> /dev/null; then
        print_error "NVIDIA driver not loaded. Please reboot and run: ./install.sh --resume"
        exit 1
    fi

    # Continue with rest of installation
    install_cuda
    install_dependencies
    setup_ai_upscale
    download_models
    copy_scripts
    optimize_gpu
    create_test_script
    show_completion
}

# Run main installation
main "$@"
