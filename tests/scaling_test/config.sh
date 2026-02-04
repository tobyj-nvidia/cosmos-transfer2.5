#!/bin/bash
# Shared configuration for all Cosmos generation scripts
# Update prompts here and re-run scripts to regenerate all outputs

# =============================================================================
# PROMPTS
# =============================================================================

# Base prompt (used for all generations)
export BASE_PROMPT="Photorealistic Kuka robot arms with dexterous Allegro hands, solid brushed metal finish, manipulating glossy blue cubes, industrial factory environment, professional studio lighting, 8K detail"

# Grid descriptions appended to prompt based on tile count
declare -A GRID_DESC
GRID_DESC[1]="single robot"
GRID_DESC[2]="1x2 grid of 2 robots"
GRID_DESC[4]="2x2 grid of 4 robots"
GRID_DESC[8]="4x2 grid of 8 robots"
GRID_DESC[16]="4x4 grid of 16 robots"
GRID_DESC[32]="8x4 grid of 32 robots"
GRID_DESC[64]="8x8 grid of 64 robots"
GRID_DESC[128]="16x8 grid of 128 robots"
GRID_DESC[256]="16x16 grid of 256 robots"
GRID_DESC[512]="32x16 grid of 512 robots"
export GRID_DESC

# =============================================================================
# COSMOS SETTINGS
# =============================================================================

export GUIDANCE=7
export NUM_STEPS=50
export CONTROL_WEIGHT=1.0
export SEG_WEIGHT=0.5

# =============================================================================
# PATHS
# =============================================================================

export STYLE_IMAGE="$HOME/cosmos/cosmos-transfer2.5/tests/scaling_test/style_reference.png"
export COSMOS_DIR="$HOME/cosmos/cosmos-transfer2.5"

# =============================================================================
# HELPER FUNCTION
# =============================================================================

# Get full prompt for a given tile count
get_prompt() {
    local tile_num=$1
    local prompt="$BASE_PROMPT"
    if [[ -n "${GRID_DESC[$tile_num]}" ]]; then
        prompt="$prompt, ${GRID_DESC[$tile_num]}"
    fi
    echo "$prompt"
}
export -f get_prompt

