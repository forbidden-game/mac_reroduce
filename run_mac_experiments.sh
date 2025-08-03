#!/bin/bash

# MAC Comprehensive Experiments Launcher
# A simple script to set up and run the complete MAC experiment pipeline

set -e  # Exit on any error

echo "🚀 MAC Comprehensive Experiments Launcher"
echo "=========================================="

# Function to print colored output
print_status() {
    echo -e "\033[1;34m[INFO]\033[0m $1"
}

print_success() {
    echo -e "\033[1;32m[SUCCESS]\033[0m $1"
}

print_error() {
    echo -e "\033[1;31m[ERROR]\033[0m $1"
}

print_warning() {
    echo -e "\033[1;33m[WARNING]\033[0m $1"
}

# Global variables for Python execution
PYTHON_CMD="python3"
USE_UV=false

# Check uv environment
check_uv_environment() {
    print_status "Checking uv environment..."
    
    if command -v uv &> /dev/null; then
        print_success "uv found"
        
        # Check if virtual environment is already active
        if [[ -n "$VIRTUAL_ENV" ]]; then
            print_success "Virtual environment already active: $VIRTUAL_ENV"
            
            # Check if it's a uv-managed environment
            if [[ -f "$VIRTUAL_ENV/pyvenv.cfg" ]]; then
                print_success "Using active uv virtual environment"
                PYTHON_CMD="$VIRTUAL_ENV/bin/python"
                USE_UV=true
                return 0
            fi
        elif [[ -d ".venv" ]]; then
            print_status "Found .venv directory - activating automatically..."
            
            # Force activate .venv environment
            export VIRTUAL_ENV="$(pwd)/.venv"
            export PATH="$VIRTUAL_ENV/bin:$PATH"
            export PYTHONPATH="$VIRTUAL_ENV/lib/python*/site-packages:$PYTHONPATH"
            
            # Set Python command to use .venv directly
            PYTHON_CMD="$VIRTUAL_ENV/bin/python"
            USE_UV=true
            
            print_success "Activated .venv environment: $VIRTUAL_ENV"
            print_success "Using Python: $PYTHON_CMD"
            return 0
        fi
        
        print_warning "uv found but no virtual environment detected"
        print_status "Will use uv for package management but regular Python execution"
    else
        print_status "uv not found, using standard Python environment"
    fi
    
    USE_UV=false
    PYTHON_CMD="python3"
}

# Check Python version
check_python() {
    print_status "Checking Python version..."
    
    if $USE_UV; then
        if $PYTHON_CMD -c "import sys" &> /dev/null; then
            PYTHON_VERSION=$($PYTHON_CMD -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
            print_success "Python $PYTHON_VERSION found (via uv)"
        else
            print_error "Python not accessible via uv. Please check your environment."
            exit 1
        fi
    else
        if command -v python3 &> /dev/null; then
            PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
            print_success "Python $PYTHON_VERSION found"
        else
            print_error "Python3 not found. Please install Python 3.8+"
            exit 1
        fi
    fi
    
    # Check if version is >= 3.8
    if $PYTHON_CMD -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)"; then
        print_success "Python version is compatible"
    else
        print_error "Python 3.8+ required, found $PYTHON_VERSION"
        exit 1
    fi
}

# Check CUDA availability
check_cuda() {
    print_status "Checking CUDA availability..."
    if command -v nvidia-smi &> /dev/null; then
        print_success "NVIDIA GPU detected"
        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits | while read gpu; do
            print_success "GPU: $gpu"
        done
    else
        print_warning "NVIDIA GPU not detected. Training will use CPU (much slower)"
    fi
}

# Install dependencies
install_dependencies() {
    print_status "Installing dependencies..."
    
    if $USE_UV; then
        print_status "Using uv for fast package installation..."
        print_status "Target environment: $VIRTUAL_ENV"
        
        # Install PyTorch with CUDA 12.8 first (required for this GPU setup)
        print_status "Installing PyTorch with CUDA 12.8 support to .venv..."
        if uv pip install --python "$PYTHON_CMD" torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128; then
            print_success "PyTorch with CUDA 12.8 installed successfully to .venv"
        else
            print_error "Failed to install PyTorch with CUDA 12.8"
            exit 1
        fi
        
        # Install other dependencies from pyproject.toml (but skip sync since it removes PyTorch)
        if [[ -f "pyproject.toml" ]] && grep -q "dependencies" pyproject.toml 2>/dev/null; then
            print_status "Installing other dependencies from pyproject.toml to .venv..."
            print_warning "Skipping uv sync to preserve PyTorch CUDA installation"
            # Install remaining dependencies excluding torch (already installed)
            # Fix protobuf version for tensorboard-logger compatibility
            uv pip install --python "$PYTHON_CMD" "protobuf<=3.20.3" numpy scipy scikit-learn matplotlib seaborn PyYAML tqdm psutil plotly h5py tensorboard tensorboard-logger PyWavelets
            print_success "Base dependencies installed to .venv"
        else
            print_status "Installing remaining dependencies to .venv..."
            # Install everything except torch from requirements (torch already installed with CUDA)
            uv pip install --python "$PYTHON_CMD" numpy pandas scikit-learn scipy matplotlib seaborn plotly PyYAML tqdm psutil
        fi
    else
        print_status "Using pip for package installation..."
        
        # Install PyTorch with CUDA 12.8 first
        print_status "Installing PyTorch with CUDA 12.8 support..."
        if pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128; then
            print_success "PyTorch with CUDA 12.8 installed successfully"
        else
            print_error "Failed to install PyTorch with CUDA 12.8"
            exit 1
        fi
        
        # Install other dependencies
        pip install numpy pandas scikit-learn scipy matplotlib seaborn plotly PyYAML tqdm psutil
    fi
    
    # Verify PyTorch CUDA installation
    print_status "Verifying PyTorch CUDA installation..."
    print_status "Using Python: $PYTHON_CMD"
    if $PYTHON_CMD -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')"; then
        print_success "PyTorch CUDA verification successful"
    else
        print_error "PyTorch CUDA verification failed"
        print_error "Python path: $PYTHON_CMD"
        print_error "Virtual env: $VIRTUAL_ENV"
        exit 1
    fi
    
    print_success "All dependencies installed successfully"
}

# Verify datasets
verify_datasets() {
    print_status "Verifying dataset preprocessing..."
    
    if $PYTHON_CMD verify_corrected_splits.py; then
        print_success "Datasets verified and ready"
    else
        print_error "Dataset verification failed"
        print_status "Attempting to preprocess datasets..."
        
        # Try to preprocess datasets (note: files were renamed during cleanup)
        $PYTHON_CMD preprocess_rml2016a.py
        $PYTHON_CMD preprocess_rml2016b.py
        $PYTHON_CMD preprocess_rml2018_memory_efficient.py
        
        # Verify again
        if $PYTHON_CMD verify_corrected_splits.py; then
            print_success "Datasets preprocessed and verified"
        else
            print_error "Failed to preprocess datasets. Please check data files."
            exit 1
        fi
    fi
}

# Display experiment overview
show_overview() {
    echo ""
    echo "📊 EXPERIMENT OVERVIEW"
    echo "======================"
    echo "Total Experiments: 66 training runs"
    echo "  • RML201610A: 20 SNR levels (240 epochs each)"
    echo "  • RML201610B: 20 SNR levels (240 epochs each)"
    echo "  • RML2018: 26 SNR levels (120 epochs each)"
    echo ""
    echo "Estimated Time: ~26 hours total"
    echo "Output Directory: comprehensive_mac_experiments/"
    echo ""
}

# Main experiment options
show_menu() {
    echo "🎯 EXPERIMENT OPTIONS"
    echo "===================="
    echo "1. Run ALL experiments (66 runs, ~26 hours)"
    echo "2. Run RML201610A only (20 runs, ~8 hours)"
    echo "3. Run RML201610B only (20 runs, ~8 hours)"
    echo "4. Run RML2018 only (26 runs, ~10 hours)"
    echo "5. Custom configuration"
    echo "6. Resume previous run"
    echo "7. Analyze existing results"
    echo "8. Generate final report"
    echo "9. Quick test (few experiments)"
    echo "0. Exit"
    echo ""
    read -p "Select option [0-9]: " choice
}

# Run experiments based on choice
run_experiments() {
    case $1 in
        1)
            print_status "Starting ALL experiments..."
            $PYTHON_CMD run_comprehensive_experiments.py
            ;;
        2)
            print_status "Starting RML201610A experiments..."
            $PYTHON_CMD run_comprehensive_experiments.py --datasets RML201610A
            ;;
        3)
            print_status "Starting RML201610B experiments..."
            $PYTHON_CMD run_comprehensive_experiments.py --datasets RML201610B
            ;;
        4)
            print_status "Starting RML2018 experiments..."
            $PYTHON_CMD run_comprehensive_experiments.py --datasets RML2018
            ;;
        5)
            print_status "Using custom configuration..."
            read -p "Enter config file path [automation_config.yaml]: " config_file
            config_file=${config_file:-automation_config.yaml}
            $PYTHON_CMD run_comprehensive_experiments.py --config "$config_file"
            ;;
        6)
            print_status "Resuming previous experiments..."
            $PYTHON_CMD run_comprehensive_experiments.py --resume
            ;;
        7)
            print_status "Analyzing existing results..."
            $PYTHON_CMD analyze_results.py
            ;;
        8)
            print_status "Generating final report..."
            $PYTHON_CMD generate_final_report.py
            ;;
        9)
            print_status "Running quick test..."
            # Create a quick test config
            cat > quick_test_config.yaml << EOF
experiment_settings:
  name: "MAC_Quick_Test"
  description: "Quick test of MAC training"
  output_dir: "quick_test_results"
  log_level: "INFO"

datasets:
  RML201610A:
    enabled: true
    epochs: 10
    batch_size: 64
    snr_levels: [0, 10]
    num_classes: 11
    class_names: ['8PSK', 'AM-DSB', 'AM-SSB', 'BPSK', 'CPFSK', 'GFSK', 'PAM4', 'QAM16', 'QAM64', 'QPSK', 'WBFM']
    
  RML201610B:
    enabled: false
    
  RML2018:
    enabled: false

training:
  learning_rate: 0.01
  lr_decay_epochs: [8]
  lr_decay_rate: 0.5
  momentum: 0.9
  weight_decay: 0.0001
  nce_k: 4096
  nce_t: 0.07
  nce_m: 0.9
  view_chose: "ALL"
  mod_l: "AN"
  n_1: 1.0
  n_t: 1.0
  feat_dim: 128
  num_workers: 4
  print_freq: 5
  save_freq: 10

automation:
  sequential_execution: true
  cleanup_between_runs: true
  gpu_memory_threshold: 0.9
  max_retries: 2
  resume_on_failure: true
  save_intermediate_results: true
  generate_progress_plots: true

analysis:
  generate_tsne: false
  generate_comparison_plots: true
  statistical_analysis: true
  cross_dataset_evaluation: false
  convergence_analysis: true
  snr_sensitivity_analysis: true

reporting:
  generate_html_report: true
  generate_pdf_summary: false
  include_training_curves: true
  include_feature_visualizations: false
  include_statistical_tests: false
EOF
            $PYTHON_CMD run_comprehensive_experiments.py --config quick_test_config.yaml
            ;;
        0)
            print_status "Exiting..."
            exit 0
            ;;
        *)
            print_error "Invalid option. Please try again."
            ;;
    esac
}

# Post-experiment actions
post_experiment_menu() {
    echo ""
    echo "✅ Experiments completed!"
    echo ""
    echo "📊 NEXT STEPS"
    echo "============="
    echo "1. Analyze results"
    echo "2. Generate final report"
    echo "3. View results directory"
    echo "4. Return to main menu"
    echo "0. Exit"
    echo ""
    read -p "Select option [0-4]: " post_choice
    
    case $post_choice in
        1)
            print_status "Analyzing results..."
            $PYTHON_CMD analyze_results.py
            post_experiment_menu
            ;;
        2)
            print_status "Generating final report..."
            $PYTHON_CMD generate_final_report.py
            if [ -f "comprehensive_mac_experiments/final_report/comprehensive_report.html" ]; then
                echo ""
                print_success "Report generated successfully!"
                echo "📄 Open: comprehensive_mac_experiments/final_report/comprehensive_report.html"
                
                # Try to open the report in browser (Linux/Mac)
                if command -v xdg-open &> /dev/null; then
                    read -p "Open report in browser? [y/N]: " open_browser
                    if [[ $open_browser =~ ^[Yy]$ ]]; then
                        xdg-open "comprehensive_mac_experiments/final_report/comprehensive_report.html"
                    fi
                elif command -v open &> /dev/null; then
                    read -p "Open report in browser? [y/N]: " open_browser
                    if [[ $open_browser =~ ^[Yy]$ ]]; then
                        open "comprehensive_mac_experiments/final_report/comprehensive_report.html"
                    fi
                fi
            fi
            post_experiment_menu
            ;;
        3)
            if [ -d "comprehensive_mac_experiments" ]; then
                print_status "Results directory: $(pwd)/comprehensive_mac_experiments"
                ls -la comprehensive_mac_experiments/
            else
                print_warning "Results directory not found"
            fi
            post_experiment_menu
            ;;
        4)
            main_loop
            ;;
        0)
            print_status "Exiting..."
            exit 0
            ;;
        *)
            print_error "Invalid option. Please try again."
            post_experiment_menu
            ;;
    esac
}

# Main execution loop
main_loop() {
    while true; do
        show_overview
        show_menu
        run_experiments $choice
        
        # If experiments were run (not analysis/report), show post-experiment menu
        if [[ $choice =~ ^[1-6,9]$ ]]; then
            post_experiment_menu
        fi
    done
}

# Main execution
main() {
    echo ""
    print_status "Initializing MAC Comprehensive Experiments..."
    
    # System checks
    check_uv_environment
    check_python
    check_cuda
    
    # Setup
    print_status "Setting up environment..."
    install_dependencies
    verify_datasets
    
    print_success "Setup completed successfully!"
    
    # Start main loop
    main_loop
}

# Handle Ctrl+C gracefully
trap 'echo -e "\n\n⏹️  Interrupted by user. Exiting..."; exit 130' INT

# Run main function
main "$@"
