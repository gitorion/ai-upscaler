#!/bin/bash

##############################################################################
# Test Script - Verify Model Download URLs
# Run this before installation to test internet connectivity and URLs
##############################################################################

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

print_success() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

test_url() {
    local name="$1"
    local url="$2"

    printf "Testing: %-40s " "$name"

    # Test the URL
    if curl -s -I -L "$url" | grep -q "HTTP.* 200"; then
        echo -e "${GREEN}✓ OK${NC}"
        return 0
    else
        echo -e "${RED}✗ FAILED${NC}"
        echo "  URL: $url"
        return 1
    fi
}

print_header "Real-ESRGAN Model URL Test"

print_info "Testing internet connectivity and model download URLs..."
print_info "This helps verify the installation will succeed before running it"
echo ""

# Test GitHub connectivity first
print_info "Testing GitHub connectivity..."
if curl -s -I https://github.com | grep -q "HTTP.* 200\|HTTP.* 301"; then
    print_success "GitHub is reachable"
else
    print_error "Cannot reach GitHub - check your internet connection"
    exit 1
fi

echo ""
print_info "Testing model download URLs..."
echo ""

failed_count=0

# Test all model URLs
test_url "RealESRGAN_x4plus.pth" \
    "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth" || ((failed_count++))

test_url "RealESRGAN_x2plus.pth" \
    "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth" || ((failed_count++))

test_url "RealESRGAN_x4plus_anime_6B.pth" \
    "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth" || ((failed_count++))

test_url "RealESRNet_x4plus.pth" \
    "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth" || ((failed_count++))

echo ""

# Summary
if [ $failed_count -eq 0 ]; then
    print_success "All model URLs are accessible!"
    print_info "You can proceed with the installation"
    echo ""
    print_info "Run: ./install.sh"
else
    print_error "$failed_count model URL(s) failed"
    print_info "Possible issues:"
    echo "  - Internet connection problems"
    echo "  - GitHub is blocking your IP"
    echo "  - Model URLs have changed"
    echo ""
    print_info "You can still try the installation - it will retry failed downloads"
    echo "  Or download models manually from: https://github.com/xinntao/Real-ESRGAN/releases"
fi

echo ""
