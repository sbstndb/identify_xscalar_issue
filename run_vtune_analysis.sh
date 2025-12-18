#!/bin/bash
#
# run_vtune_analysis.sh - Comprehensive VTune profiling for xtensor benchmarks
#
# USAGE:
#   ./run_vtune_analysis.sh collect [OPTIONS]    # Collect VTune data (exhaustive)
#   ./run_vtune_analysis.sh report [OPTIONS]     # Generate text reports from existing data
#   ./run_vtune_analysis.sh asm [OPTIONS]        # Extract assembly for benchmarks
#   ./run_vtune_analysis.sh compare [OPTIONS]    # Generate comparisons between configs
#   ./run_vtune_analysis.sh list                 # List available configs and benchmarks
#   ./run_vtune_analysis.sh all [OPTIONS]        # Run collect + report + asm
#
# OPTIONS:
#   --configs=LIST         Comma-separated configs (default: all)
#                          Example: --configs=gcc14_xsimd,gcc14_noxsimd
#   --benchmarks=FILTER    Regex filter for benchmarks (default: all double)
#                          Example: --benchmarks="fixed_double_(16|32)"
#   --duration=SECS        Min benchmark duration in seconds (default: 3)
#   --analysis=TYPE        VTune analysis type: hotspots,microarch,memory (default: hotspots)
#   --output-dir=DIR       Output directory (default: ./analysis_results)
#   --resume               Skip existing results (default: true)
#   --no-resume            Force re-collection of all results
#   --parallel=N           Number of parallel collections (default: 1)
#   --compare-pairs=SPEC   Comparison pairs, e.g., "gcc14_xsimd:gcc14_noxsimd,gcc13:gcc14"
#
# EXAMPLES:
#   # Collect all benchmarks (exhaustive)
#   ./run_vtune_analysis.sh collect
#
#   # Collect only fixed benchmarks for gcc14
#   ./run_vtune_analysis.sh collect --configs=gcc14_xsimd,gcc14_noxsimd --benchmarks="fixed_double"
#
#   # Generate comparison between xsimd and noxsimd
#   ./run_vtune_analysis.sh compare --compare-pairs="gcc14_xsimd:gcc14_noxsimd"
#
#   # Extract assembly for worst offenders
#   ./run_vtune_analysis.sh asm --configs=gcc13_noxsimd --benchmarks="fixed_double_1024"
#

set -uo pipefail
# Note: Not using -e because some commands may legitimately return non-zero

# ==============================================================================
# CONFIGURATION
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASEDIR="${SCRIPT_DIR}"
DEFAULT_OUTPUT_DIR="${BASEDIR}/analysis_results"

# VTune setup
VTUNE_DIR="/opt/intel/oneapi/vtune/2025.4"
VTUNE_BIN="${VTUNE_DIR}/bin64/vtune"
VTUNE_VARS="${VTUNE_DIR}/vtune-vars.sh"

# Default settings
DEFAULT_DURATION=3
DEFAULT_ANALYSIS="hotspots"
RESUME=true
PARALLEL=1

# All available configurations
ALL_CONFIGS="gcc11_xsimd gcc11_noxsimd gcc12_xsimd gcc12_noxsimd gcc13_xsimd gcc13_noxsimd gcc14_xsimd gcc14_noxsimd clang16_xsimd clang16_noxsimd clang17_xsimd clang17_noxsimd clang18_xsimd clang18_noxsimd clang19_xsimd clang19_noxsimd clang20_xsimd clang20_noxsimd"

# All benchmark names (will be populated from executable)
ALL_BENCHMARKS=""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================

log_info() {
    echo -e "${BLUE}[INFO]${NC} $*"
}

log_success() {
    echo -e "${GREEN}[OK]${NC} $*"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $*"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $*" >&2
}

check_vtune() {
    if [[ ! -x "${VTUNE_BIN}" ]]; then
        log_error "VTune not found at ${VTUNE_BIN}"
        log_info "Please ensure Intel VTune is installed"
        exit 1
    fi
    log_info "Using VTune: ${VTUNE_BIN}"
}

get_benchmark_list() {
    local config=$1
    local exe="${BASEDIR}/build_${config}/bench_xscalar"

    if [[ -x "${exe}" ]]; then
        "${exe}" --benchmark_list_tests=true 2>/dev/null | grep -E "^(xtensor|fixed)_double"
    fi
}

filter_benchmarks() {
    local filter=$1
    if [[ -z "${filter}" ]] || [[ "${filter}" == "all" ]]; then
        cat  # Pass through all
    else
        grep -E "${filter}" || true
    fi
}

ensure_output_dir() {
    local dir=$1
    mkdir -p "${dir}"
}

result_exists() {
    local result_dir=$1
    # Check for VTune result directory structure
    # Can be: data.0/*.trace, r000hs/, or sqlite-db/
    [[ -d "${result_dir}" ]] && [[ -d "${result_dir}/sqlite-db" || -d "${result_dir}/data.0" || -d "${result_dir}/r000hs" ]]
}

# ==============================================================================
# COLLECTION FUNCTIONS
# ==============================================================================

collect_vtune_single() {
    local config=$1
    local benchmark=$2
    local analysis=$3
    local duration=$4
    local output_dir=$5

    local exe="${BASEDIR}/build_${config}/bench_xscalar"
    local result_dir="${output_dir}/vtune_raw/${config}/${benchmark}_${analysis}"

    # Check if executable exists
    if [[ ! -x "${exe}" ]]; then
        log_warn "Executable not found: ${exe}"
        return 1
    fi

    # Check if result already exists (resume mode)
    if [[ "${RESUME}" == "true" ]] && result_exists "${result_dir}"; then
        log_info "Skipping (exists): ${config}/${benchmark}"
        return 0
    fi

    ensure_output_dir "$(dirname "${result_dir}")"

    # Remove old result if exists
    [[ -d "${result_dir}" ]] && rm -rf "${result_dir}"

    log_info "Collecting: ${config}/${benchmark} (${analysis})"

    # Map analysis type to VTune collection
    local collect_type=""
    local extra_knobs=""
    case "${analysis}" in
        hotspots)
            collect_type="hotspots"
            # Use software sampling (works without driver on hybrid CPUs)
            extra_knobs="-knob sampling-mode=sw -knob enable-stack-collection=true"
            ;;
        microarch)
            collect_type="uarch-exploration"
            extra_knobs="-knob sampling-mode=sw"
            ;;
        memory)
            collect_type="memory-access"
            extra_knobs="-knob sampling-mode=sw"
            ;;
        *)
            collect_type="hotspots"
            extra_knobs="-knob sampling-mode=sw -knob enable-stack-collection=true"
            ;;
    esac

    # Run VTune collection
    # Note: Using software sampling mode (sw) for compatibility with hybrid CPUs
    # without requiring the VTune sampling driver
    if "${VTUNE_BIN}" -collect "${collect_type}" \
            ${extra_knobs} \
            -result-dir="${result_dir}" \
            -- "${exe}" \
               --benchmark_filter="^${benchmark}$" \
               --benchmark_min_time="${duration}s"; then
        log_success "Collected: ${config}/${benchmark}"
        return 0
    else
        log_error "Failed: ${config}/${benchmark}"
        return 1
    fi
}

collect_all() {
    local configs=$1
    local benchmark_filter=$2
    local analysis=$3
    local duration=$4
    local output_dir=$5

    local total=0
    local success=0
    local failed=0
    local skipped=0

    log_info "Starting exhaustive VTune collection..."
    log_info "Configs: ${configs}"
    log_info "Benchmark filter: ${benchmark_filter:-all}"
    log_info "Analysis type: ${analysis}"
    log_info "Duration: ${duration}s per benchmark"
    log_info "Output: ${output_dir}"
    echo ""

    # Create progress tracking file
    local progress_file="${output_dir}/.collection_progress"
    ensure_output_dir "${output_dir}"

    for config in ${configs}; do
        local exe="${BASEDIR}/build_${config}/bench_xscalar"

        if [[ ! -x "${exe}" ]]; then
            log_warn "Skipping config ${config}: executable not found"
            continue
        fi

        # Get benchmarks for this config
        local benchmarks
        benchmarks=$(get_benchmark_list "${config}" | filter_benchmarks "${benchmark_filter}")

        if [[ -z "${benchmarks}" ]]; then
            log_warn "No benchmarks matching filter for ${config}"
            continue
        fi

        for benchmark in ${benchmarks}; do
            total=$((total + 1))

            if collect_vtune_single "${config}" "${benchmark}" "${analysis}" "${duration}" "${output_dir}"; then
                if result_exists "${output_dir}/vtune_raw/${config}/${benchmark}_${analysis}"; then
                    success=$((success + 1))
                else
                    skipped=$((skipped + 1))
                fi
            else
                failed=$((failed + 1))
            fi

            # Update progress
            echo "${config}/${benchmark}: $(date)" >> "${progress_file}"
        done
    done

    echo ""
    log_info "Collection complete!"
    log_info "Total: ${total}, Success: ${success}, Skipped: ${skipped}, Failed: ${failed}"
}

# ==============================================================================
# REPORT GENERATION FUNCTIONS
# ==============================================================================

generate_report_single() {
    local config=$1
    local benchmark=$2
    local analysis=$3
    local output_dir=$4

    local result_dir="${output_dir}/vtune_raw/${config}/${benchmark}_${analysis}"
    local report_dir="${output_dir}/reports/hotspots"
    local report_file="${report_dir}/${config}_${benchmark}.txt"

    if ! result_exists "${result_dir}"; then
        log_warn "No VTune data for ${config}/${benchmark}"
        return 1
    fi

    ensure_output_dir "${report_dir}"

    log_info "Generating report: ${config}/${benchmark}"

    # Generate hotspots report
    {
        echo "================================================================================"
        echo "VTUNE HOTSPOTS REPORT"
        echo "================================================================================"
        echo "Configuration: ${config}"
        echo "Benchmark:     ${benchmark}"
        echo "Analysis:      ${analysis}"
        echo "Generated:     $(date)"
        echo "================================================================================"
        echo ""

        "${VTUNE_BIN}" -report hotspots \
            -result-dir="${result_dir}" \
            -format=text \
            -report-width=200 \
            2>/dev/null || echo "[Report generation failed]"

        echo ""
        echo "================================================================================"
        echo "TOP-DOWN / SUMMARY"
        echo "================================================================================"

        "${VTUNE_BIN}" -report summary \
            -result-dir="${result_dir}" \
            -format=text \
            2>/dev/null || echo "[Summary not available]"

    } > "${report_file}"

    log_success "Report: ${report_file}"
}

generate_all_reports() {
    local configs=$1
    local benchmark_filter=$2
    local analysis=$3
    local output_dir=$4

    log_info "Generating text reports..."

    for config in ${configs}; do
        local vtune_config_dir="${output_dir}/vtune_raw/${config}"

        if [[ ! -d "${vtune_config_dir}" ]]; then
            continue
        fi

        # Find all collected benchmarks for this config
        for result_path in "${vtune_config_dir}"/*_"${analysis}"; do
            if [[ -d "${result_path}" ]]; then
                local benchmark
                benchmark=$(basename "${result_path}" | sed "s/_${analysis}$//")

                # Apply filter
                if [[ -n "${benchmark_filter}" ]] && ! echo "${benchmark}" | grep -qE "${benchmark_filter}"; then
                    continue
                fi

                generate_report_single "${config}" "${benchmark}" "${analysis}" "${output_dir}"
            fi
        done
    done

    log_success "All reports generated in ${output_dir}/reports/"
}

# ==============================================================================
# ASSEMBLY EXTRACTION FUNCTIONS
# ==============================================================================

extract_asm_single() {
    local config=$1
    local benchmark=$2
    local output_dir=$3

    local exe="${BASEDIR}/build_${config}/bench_xscalar"
    local asm_dir="${output_dir}/asm/${config}"
    local asm_file="${asm_dir}/${benchmark}.asm"

    if [[ ! -x "${exe}" ]]; then
        log_warn "Executable not found: ${exe}"
        return 1
    fi

    ensure_output_dir "${asm_dir}"

    log_info "Extracting ASM: ${config}/${benchmark}"

    # Determine the function name pattern based on benchmark
    local func_pattern=""
    if [[ "${benchmark}" == fixed_* ]]; then
        # fixed_double_16 -> BM_XScalarAdd_fixed<double, 16ul>
        local size
        size=$(echo "${benchmark}" | sed 's/fixed_double_//')
        func_pattern="BM_XScalarAdd_fixed.*double.*${size}"
    else
        # xtensor_double_16 -> BM_XScalarAdd_xtensor<double, 16ul>
        local size
        size=$(echo "${benchmark}" | sed 's/xtensor_double_//')
        func_pattern="BM_XScalarAdd_xtensor.*double.*${size}"
    fi

    {
        echo "================================================================================"
        echo "DISASSEMBLY"
        echo "================================================================================"
        echo "Configuration: ${config}"
        echo "Benchmark:     ${benchmark}"
        echo "Function:      ${func_pattern}"
        echo "Executable:    ${exe}"
        echo "Generated:     $(date)"
        echo "================================================================================"
        echo ""

        # Try to get function address and disassemble
        local func_info
        func_info=$(nm "${exe}" 2>/dev/null | c++filt | grep -E "${func_pattern}" | head -1)

        if [[ -n "${func_info}" ]]; then
            local addr
            addr=$(echo "${func_info}" | awk '{print $1}')
            local func_name
            func_name=$(echo "${func_info}" | awk '{print $3}')

            echo "Symbol: ${func_name}"
            echo "Address: 0x${addr}"
            echo ""
            echo "--------------------------------------------------------------------------------"
            echo ""

            # Disassemble with source interleaved
            objdump -d -S -M intel --no-show-raw-insn "${exe}" 2>/dev/null | \
                c++filt | \
                awk -v pattern="${func_pattern}" '
                    BEGIN { found=0 }
                    /^[0-9a-f]+ <.*'"${func_pattern}"'.*>:/ { found=1 }
                    found && /^[0-9a-f]+ <[^>]+>:/ && !/'"${func_pattern}"'/ { found=0 }
                    found { print }
                ' | head -500
        else
            echo "[Function not found in symbol table]"
            echo "Attempting full disassembly grep..."
            echo ""

            objdump -d -M intel "${exe}" 2>/dev/null | \
                c++filt | \
                grep -A 100 "${func_pattern}" | head -150
        fi

    } > "${asm_file}"

    log_success "ASM: ${asm_file}"
}

extract_all_asm() {
    local configs=$1
    local benchmark_filter=$2
    local output_dir=$3

    log_info "Extracting assembly..."

    for config in ${configs}; do
        local exe="${BASEDIR}/build_${config}/bench_xscalar"

        if [[ ! -x "${exe}" ]]; then
            continue
        fi

        # Get benchmarks for this config
        local benchmarks
        benchmarks=$(get_benchmark_list "${config}" | filter_benchmarks "${benchmark_filter}")

        for benchmark in ${benchmarks}; do
            extract_asm_single "${config}" "${benchmark}" "${output_dir}"
        done
    done

    log_success "All assembly extracted to ${output_dir}/asm/"
}

# ==============================================================================
# COMPARISON FUNCTIONS
# ==============================================================================

compare_vtune_pair() {
    local config1=$1
    local config2=$2
    local benchmark=$3
    local analysis=$4
    local output_dir=$5

    local result1="${output_dir}/vtune_raw/${config1}/${benchmark}_${analysis}"
    local result2="${output_dir}/vtune_raw/${config2}/${benchmark}_${analysis}"
    local compare_dir="${output_dir}/comparisons/${benchmark}"
    local compare_file="${compare_dir}/${config1}_vs_${config2}.txt"

    if ! result_exists "${result1}" || ! result_exists "${result2}"; then
        log_warn "Missing VTune data for comparison: ${config1} or ${config2} / ${benchmark}"
        return 1
    fi

    ensure_output_dir "${compare_dir}"

    log_info "Comparing: ${config1} vs ${config2} for ${benchmark}"

    {
        echo "================================================================================"
        echo "VTUNE COMPARISON REPORT"
        echo "================================================================================"
        echo "Benchmark:     ${benchmark}"
        echo "Config A:      ${config1}"
        echo "Config B:      ${config2}"
        echo "Generated:     $(date)"
        echo "================================================================================"
        echo ""
        echo "================================================================================"
        echo "CONFIG A: ${config1}"
        echo "================================================================================"
        echo ""

        "${VTUNE_BIN}" -report hotspots \
            -result-dir="${result1}" \
            -format=text \
            -report-width=180 \
            2>/dev/null || echo "[Report A failed]"

        echo ""
        echo "================================================================================"
        echo "CONFIG B: ${config2}"
        echo "================================================================================"
        echo ""

        "${VTUNE_BIN}" -report hotspots \
            -result-dir="${result2}" \
            -format=text \
            -report-width=180 \
            2>/dev/null || echo "[Report B failed]"

        echo ""
        echo "================================================================================"
        echo "SUMMARY COMPARISON"
        echo "================================================================================"
        echo ""
        echo "Top functions unique to ${config1}:"
        comm -23 \
            <("${VTUNE_BIN}" -report hotspots -result-dir="${result1}" -format=csv 2>/dev/null | tail -n +2 | cut -d',' -f1 | sort -u) \
            <("${VTUNE_BIN}" -report hotspots -result-dir="${result2}" -format=csv 2>/dev/null | tail -n +2 | cut -d',' -f1 | sort -u) \
            2>/dev/null | head -10 | sed 's/^/  - /'

        echo ""
        echo "Top functions unique to ${config2}:"
        comm -13 \
            <("${VTUNE_BIN}" -report hotspots -result-dir="${result1}" -format=csv 2>/dev/null | tail -n +2 | cut -d',' -f1 | sort -u) \
            <("${VTUNE_BIN}" -report hotspots -result-dir="${result2}" -format=csv 2>/dev/null | tail -n +2 | cut -d',' -f1 | sort -u) \
            2>/dev/null | head -10 | sed 's/^/  - /'

    } > "${compare_file}"

    log_success "Comparison: ${compare_file}"
}

compare_asm_pair() {
    local config1=$1
    local config2=$2
    local benchmark=$3
    local output_dir=$4

    local asm1="${output_dir}/asm/${config1}/${benchmark}.asm"
    local asm2="${output_dir}/asm/${config2}/${benchmark}.asm"
    local compare_dir="${output_dir}/comparisons/${benchmark}"
    local diff_file="${compare_dir}/${config1}_vs_${config2}_asm.diff"

    if [[ ! -f "${asm1}" ]] || [[ ! -f "${asm2}" ]]; then
        log_warn "Missing ASM files for comparison"
        return 1
    fi

    ensure_output_dir "${compare_dir}"

    log_info "Comparing ASM: ${config1} vs ${config2} for ${benchmark}"

    # Create normalized versions (remove addresses)
    local tmp1 tmp2
    tmp1=$(mktemp)
    tmp2=$(mktemp)

    # Normalize: keep only instructions
    grep -E '^\s+[0-9a-f]+:' "${asm1}" | \
        sed 's/^\s*[0-9a-f]*:\s*//' > "${tmp1}"
    grep -E '^\s+[0-9a-f]+:' "${asm2}" | \
        sed 's/^\s*[0-9a-f]*:\s*//' > "${tmp2}"

    {
        echo "================================================================================"
        echo "ASM DIFF: ${config1} vs ${config2}"
        echo "Benchmark: ${benchmark}"
        echo "================================================================================"
        echo ""
        echo "< = ${config1}"
        echo "> = ${config2}"
        echo ""
        echo "--------------------------------------------------------------------------------"
        echo ""

        diff -u "${tmp1}" "${tmp2}" || true

    } > "${diff_file}"

    rm -f "${tmp1}" "${tmp2}"

    log_success "ASM diff: ${diff_file}"
}

generate_comparisons() {
    local compare_pairs=$1
    local benchmark_filter=$2
    local analysis=$3
    local output_dir=$4

    log_info "Generating comparisons..."

    # If no pairs specified, generate default comparisons
    if [[ -z "${compare_pairs}" ]]; then
        # Default: compare xsimd vs noxsimd for each compiler
        compare_pairs=""
        for compiler in gcc11 gcc12 gcc13 gcc14 clang16 clang17 clang18 clang19 clang20; do
            compare_pairs="${compare_pairs} ${compiler}_xsimd:${compiler}_noxsimd"
        done
    fi

    # Parse pairs and generate comparisons
    for pair in ${compare_pairs}; do
        local config1 config2
        config1=$(echo "${pair}" | cut -d: -f1)
        config2=$(echo "${pair}" | cut -d: -f2)

        # Find common benchmarks between the two configs
        local result_dir1="${output_dir}/vtune_raw/${config1}"

        if [[ ! -d "${result_dir1}" ]]; then
            continue
        fi

        for result_path in "${result_dir1}"/*_"${analysis}"; do
            if [[ -d "${result_path}" ]]; then
                local benchmark
                benchmark=$(basename "${result_path}" | sed "s/_${analysis}$//")

                # Apply filter
                if [[ -n "${benchmark_filter}" ]] && ! echo "${benchmark}" | grep -qE "${benchmark_filter}"; then
                    continue
                fi

                compare_vtune_pair "${config1}" "${config2}" "${benchmark}" "${analysis}" "${output_dir}"
                compare_asm_pair "${config1}" "${config2}" "${benchmark}" "${output_dir}"
            fi
        done
    done

    log_success "Comparisons generated in ${output_dir}/comparisons/"
}

# ==============================================================================
# INDEX GENERATION
# ==============================================================================

generate_index() {
    local output_dir=$1
    local index_file="${output_dir}/INDEX.md"

    log_info "Generating index..."

    {
        echo "# VTune Analysis Results"
        echo ""
        echo "Generated: $(date)"
        echo ""
        echo "## Directory Structure"
        echo ""
        echo "\`\`\`"
        echo "analysis_results/"
        echo "├── vtune_raw/          # Raw VTune results (open with VTune GUI)"
        echo "├── reports/            # Text reports (for diff/grep)"
        echo "│   └── hotspots/       # Hotspot reports per config/benchmark"
        echo "├── asm/                # Disassembly per config/benchmark"
        echo "└── comparisons/        # Cross-config comparisons"
        echo "\`\`\`"
        echo ""
        echo "## Quick Commands"
        echo ""
        echo "### Open in VTune GUI"
        echo "\`\`\`bash"
        echo "# Source VTune environment first"
        echo "source ${VTUNE_VARS}"
        echo ""
        echo "# Open a specific result"
        echo "vtune-gui ${output_dir}/vtune_raw/gcc14_xsimd/fixed_double_16_hotspots"
        echo "\`\`\`"
        echo ""
        echo "### Compare two configurations"
        echo "\`\`\`bash"
        echo "diff ${output_dir}/reports/hotspots/gcc14_xsimd_fixed_double_16.txt \\"
        echo "     ${output_dir}/reports/hotspots/gcc14_noxsimd_fixed_double_16.txt"
        echo "\`\`\`"
        echo ""
        echo "### View assembly diff"
        echo "\`\`\`bash"
        echo "cat ${output_dir}/comparisons/fixed_double_16/gcc14_xsimd_vs_gcc14_noxsimd_asm.diff"
        echo "\`\`\`"
        echo ""
        echo "## Available Results"
        echo ""

        # List VTune results
        if [[ -d "${output_dir}/vtune_raw" ]]; then
            echo "### VTune Raw Data"
            echo ""
            for config_dir in "${output_dir}/vtune_raw"/*; do
                if [[ -d "${config_dir}" ]]; then
                    local config
                    config=$(basename "${config_dir}")
                    echo "#### ${config}"
                    echo ""
                    ls -1 "${config_dir}" 2>/dev/null | head -20 | sed 's/^/- /'
                    local count
                    count=$(ls -1 "${config_dir}" 2>/dev/null | wc -l)
                    if [[ ${count} -gt 20 ]]; then
                        echo "- ... and $((count - 20)) more"
                    fi
                    echo ""
                fi
            done
        fi

        # List comparisons
        if [[ -d "${output_dir}/comparisons" ]]; then
            echo "### Comparisons"
            echo ""
            for bench_dir in "${output_dir}/comparisons"/*; do
                if [[ -d "${bench_dir}" ]]; then
                    local benchmark
                    benchmark=$(basename "${bench_dir}")
                    echo "#### ${benchmark}"
                    ls -1 "${bench_dir}" 2>/dev/null | sed 's/^/- /'
                    echo ""
                fi
            done
        fi

    } > "${index_file}"

    log_success "Index: ${index_file}"
}

# ==============================================================================
# LIST COMMAND
# ==============================================================================

cmd_list() {
    echo "Available Configurations:"
    echo "========================="
    for config in ${ALL_CONFIGS}; do
        local exe="${BASEDIR}/build_${config}/bench_xscalar"
        if [[ -x "${exe}" ]]; then
            echo "  [OK] ${config}"
        else
            echo "  [--] ${config} (not built)"
        fi
    done

    echo ""
    echo "Available Benchmarks (from gcc14_xsimd):"
    echo "========================================="
    local exe="${BASEDIR}/build_gcc14_xsimd/bench_xscalar"
    if [[ -x "${exe}" ]]; then
        "${exe}" --benchmark_list_tests=true 2>/dev/null | grep -E "^(xtensor|fixed)_double" | sed 's/^/  /'
    else
        echo "  (build gcc14_xsimd first to see benchmarks)"
    fi
}

# ==============================================================================
# MAIN ARGUMENT PARSING
# ==============================================================================

parse_args() {
    MODE=""
    CONFIGS="${ALL_CONFIGS}"
    BENCHMARK_FILTER=""
    DURATION="${DEFAULT_DURATION}"
    ANALYSIS="${DEFAULT_ANALYSIS}"
    OUTPUT_DIR="${DEFAULT_OUTPUT_DIR}"
    COMPARE_PAIRS=""

    while [[ $# -gt 0 ]]; do
        case "$1" in
            collect|report|asm|compare|list|all)
                MODE="$1"
                shift
                ;;
            --configs=*)
                CONFIGS=$(echo "${1#*=}" | tr ',' ' ')
                shift
                ;;
            --benchmarks=*)
                BENCHMARK_FILTER="${1#*=}"
                shift
                ;;
            --duration=*)
                DURATION="${1#*=}"
                shift
                ;;
            --analysis=*)
                ANALYSIS="${1#*=}"
                shift
                ;;
            --output-dir=*)
                OUTPUT_DIR="${1#*=}"
                shift
                ;;
            --resume)
                RESUME=true
                shift
                ;;
            --no-resume)
                RESUME=false
                shift
                ;;
            --parallel=*)
                PARALLEL="${1#*=}"
                shift
                ;;
            --compare-pairs=*)
                COMPARE_PAIRS=$(echo "${1#*=}" | tr ',' ' ')
                shift
                ;;
            -h|--help)
                head -60 "$0" | tail -n +2 | grep -E "^#" | sed 's/^# \?//'
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                exit 1
                ;;
        esac
    done

    if [[ -z "${MODE}" ]]; then
        log_error "No mode specified. Use: collect, report, asm, compare, list, or all"
        echo "Run with --help for usage information"
        exit 1
    fi
}

main() {
    parse_args "$@"

    case "${MODE}" in
        list)
            cmd_list
            ;;
        collect)
            check_vtune
            collect_all "${CONFIGS}" "${BENCHMARK_FILTER}" "${ANALYSIS}" "${DURATION}" "${OUTPUT_DIR}"
            generate_index "${OUTPUT_DIR}"
            ;;
        report)
            check_vtune
            generate_all_reports "${CONFIGS}" "${BENCHMARK_FILTER}" "${ANALYSIS}" "${OUTPUT_DIR}"
            generate_index "${OUTPUT_DIR}"
            ;;
        asm)
            extract_all_asm "${CONFIGS}" "${BENCHMARK_FILTER}" "${OUTPUT_DIR}"
            generate_index "${OUTPUT_DIR}"
            ;;
        compare)
            check_vtune
            generate_comparisons "${COMPARE_PAIRS}" "${BENCHMARK_FILTER}" "${ANALYSIS}" "${OUTPUT_DIR}"
            generate_index "${OUTPUT_DIR}"
            ;;
        all)
            check_vtune
            collect_all "${CONFIGS}" "${BENCHMARK_FILTER}" "${ANALYSIS}" "${DURATION}" "${OUTPUT_DIR}"
            generate_all_reports "${CONFIGS}" "${BENCHMARK_FILTER}" "${ANALYSIS}" "${OUTPUT_DIR}"
            extract_all_asm "${CONFIGS}" "${BENCHMARK_FILTER}" "${OUTPUT_DIR}"
            generate_comparisons "${COMPARE_PAIRS}" "${BENCHMARK_FILTER}" "${ANALYSIS}" "${OUTPUT_DIR}"
            generate_index "${OUTPUT_DIR}"
            ;;
    esac
}

main "$@"
