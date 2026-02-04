#!/bin/bash
# Cleanup generated data in tile_* directories
#
# Usage:
#   ./cleanup.sh                    # Interactive - asks what to clean
#   ./cleanup.sh all                # Clean everything in all tiles
#   ./cleanup.sh 8                  # Clean everything in tile_008 only
#   ./cleanup.sh all --keep-captures # Clean outputs but keep Isaac Lab captures
#   ./cleanup.sh 8 --outputs-only   # Only clean Cosmos outputs (tile_008)
#   ./cleanup.sh all --captures-only # Only clean Isaac Lab captures
#   ./cleanup.sh all --dry-run      # Show what would be deleted
#
# Options:
#   --keep-captures   Keep capture/ directories (Isaac Lab outputs)
#   --outputs-only    Only clean output/ directories (Cosmos results)
#   --captures-only   Only clean capture/ directories (Isaac Lab)
#   --specs-only      Only clean specs/ directories
#   --dry-run         Show what would be deleted without deleting

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Parse arguments
TILE_COUNT=""
KEEP_CAPTURES=false
OUTPUTS_ONLY=false
CAPTURES_ONLY=false
SPECS_ONLY=false
DRY_RUN=false

for arg in "$@"; do
    case "$arg" in
        --keep-captures)
            KEEP_CAPTURES=true
            ;;
        --outputs-only)
            OUTPUTS_ONLY=true
            ;;
        --captures-only)
            CAPTURES_ONLY=true
            ;;
        --specs-only)
            SPECS_ONLY=true
            ;;
        --dry-run)
            DRY_RUN=true
            ;;
        all)
            TILE_COUNT="all"
            ;;
        [0-9]*)
            TILE_COUNT="$arg"
            ;;
    esac
done

# Function to get size of directory
get_size() {
    if [ -d "$1" ]; then
        du -sh "$1" 2>/dev/null | cut -f1
    else
        echo "0"
    fi
}

# Function to clean a tile directory
clean_tile() {
    local tile_dir="$1"
    local tile_name=$(basename "$tile_dir")
    
    echo "Cleaning $tile_name..."
    
    # Determine what to clean
    local clean_capture=true
    local clean_specs=true
    local clean_output=true
    
    if [ "$KEEP_CAPTURES" = true ]; then
        clean_capture=false
    fi
    
    if [ "$OUTPUTS_ONLY" = true ]; then
        clean_capture=false
        clean_specs=false
    fi
    
    if [ "$CAPTURES_ONLY" = true ]; then
        clean_specs=false
        clean_output=false
    fi
    
    if [ "$SPECS_ONLY" = true ]; then
        clean_capture=false
        clean_output=false
    fi
    
    # Clean capture/
    if [ "$clean_capture" = true ] && [ -d "$tile_dir/capture" ]; then
        local size=$(get_size "$tile_dir/capture")
        if [ "$DRY_RUN" = true ]; then
            echo "  [DRY-RUN] Would delete capture/ ($size)"
        else
            rm -rf "$tile_dir/capture"
            echo "  Deleted capture/ ($size)"
        fi
    fi
    
    # Clean specs/
    if [ "$clean_specs" = true ] && [ -d "$tile_dir/specs" ]; then
        local size=$(get_size "$tile_dir/specs")
        if [ "$DRY_RUN" = true ]; then
            echo "  [DRY-RUN] Would delete specs/ ($size)"
        else
            rm -rf "$tile_dir/specs"
            echo "  Deleted specs/ ($size)"
        fi
    fi
    
    # Clean output/
    if [ "$clean_output" = true ] && [ -d "$tile_dir/output" ]; then
        local size=$(get_size "$tile_dir/output")
        if [ "$DRY_RUN" = true ]; then
            echo "  [DRY-RUN] Would delete output/ ($size)"
        else
            rm -rf "$tile_dir/output"
            echo "  Deleted output/ ($size)"
        fi
    fi
}

# Show current disk usage
show_usage() {
    echo "Current disk usage:"
    echo ""
    
    local total_capture=0
    local total_specs=0
    local total_output=0
    
    for tile_dir in "$SCRIPT_DIR"/tile_*; do
        if [ -d "$tile_dir" ]; then
            local name=$(basename "$tile_dir")
            local cap_size=$(du -sb "$tile_dir/capture" 2>/dev/null | cut -f1 || echo 0)
            local spec_size=$(du -sb "$tile_dir/specs" 2>/dev/null | cut -f1 || echo 0)
            local out_size=$(du -sb "$tile_dir/output" 2>/dev/null | cut -f1 || echo 0)
            
            total_capture=$((total_capture + cap_size))
            total_specs=$((total_specs + spec_size))
            total_output=$((total_output + out_size))
            
            local cap_h=$(get_size "$tile_dir/capture")
            local spec_h=$(get_size "$tile_dir/specs")
            local out_h=$(get_size "$tile_dir/output")
            
            printf "  %-10s  capture: %6s  specs: %6s  output: %6s\n" "$name" "$cap_h" "$spec_h" "$out_h"
        fi
    done
    
    echo ""
    echo "Totals:"
    printf "  captures (Isaac Lab): %s\n" "$(numfmt --to=iec $total_capture 2>/dev/null || echo "${total_capture}B")"
    printf "  specs:                %s\n" "$(numfmt --to=iec $total_specs 2>/dev/null || echo "${total_specs}B")"
    printf "  outputs (Cosmos):     %s\n" "$(numfmt --to=iec $total_output 2>/dev/null || echo "${total_output}B")"
    echo ""
}

# Interactive mode
interactive_mode() {
    show_usage
    
    echo "What would you like to clean?"
    echo "  1) Everything (captures + specs + outputs)"
    echo "  2) Cosmos outputs only (keep captures)"
    echo "  3) Isaac Lab captures only"
    echo "  4) Show usage and exit"
    echo "  5) Cancel"
    echo ""
    read -p "Choice [1-5]: " choice
    
    case "$choice" in
        1)
            read -p "Clean ALL tiles? [y/N]: " confirm
            if [ "$confirm" = "y" ] || [ "$confirm" = "Y" ]; then
                TILE_COUNT="all"
            else
                echo "Cancelled."
                exit 0
            fi
            ;;
        2)
            TILE_COUNT="all"
            KEEP_CAPTURES=true
            ;;
        3)
            TILE_COUNT="all"
            CAPTURES_ONLY=true
            ;;
        4)
            exit 0
            ;;
        *)
            echo "Cancelled."
            exit 0
            ;;
    esac
}

# Main
echo "=============================================="
echo "Cleanup Script"
echo "=============================================="
echo ""

# If no arguments, show interactive mode
if [ -z "$TILE_COUNT" ]; then
    interactive_mode
fi

# Determine tiles to process
if [ "$TILE_COUNT" = "all" ]; then
    TILES=$(ls -d "$SCRIPT_DIR"/tile_* 2>/dev/null | sort)
else
    tile_num=$(printf "%03d" $TILE_COUNT)
    TILES="$SCRIPT_DIR/tile_$tile_num"
    if [ ! -d "$TILES" ]; then
        echo "Error: tile_$tile_num not found"
        exit 1
    fi
fi

# Show what will be cleaned
echo "Mode:"
[ "$KEEP_CAPTURES" = true ] && echo "  - Keeping Isaac Lab captures"
[ "$OUTPUTS_ONLY" = true ] && echo "  - Cleaning Cosmos outputs only"
[ "$CAPTURES_ONLY" = true ] && echo "  - Cleaning Isaac Lab captures only"
[ "$SPECS_ONLY" = true ] && echo "  - Cleaning specs only"
[ "$DRY_RUN" = true ] && echo "  - DRY RUN (no actual deletion)"
echo ""

# Clean each tile
for tile_dir in $TILES; do
    if [ -d "$tile_dir" ]; then
        clean_tile "$tile_dir"
    fi
done

echo ""
echo "=============================================="
if [ "$DRY_RUN" = true ]; then
    echo "Dry run complete. No files were deleted."
else
    echo "Cleanup complete!"
fi
echo "=============================================="
