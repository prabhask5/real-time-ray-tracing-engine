#!/bin/bash

# Set binary path relative to project root.
BIN_PATH="bin/raytracer"
LOCAL_BIN="./$BIN_PATH"
CLOUD_BIN="/workspace/real-time-ray-tracing-engine/$BIN_PATH"

# Parse run mode.
MODE="local"
if [[ "$1" == "cloud" || "$1" == "local" ]]; then
    MODE="$1"
    shift
fi

# Forward remaining arguments to raytracer.
ARGS=("$@")

# Determine output file name.
OUTPUT_FILE="image.ppm"
STATIC_RENDER=false
CAMERA_SET=false
DEBUG_ENABLED=false
GPU_ENABLED=false

for ((i = 0; i < ${#ARGS[@]}; i++)); do
    case "${ARGS[i]}" in
        --output)
            if [[ $((i+1)) -lt ${#ARGS[@]} ]]; then
                OUTPUT_FILE="${ARGS[i+1]}"
            fi
            ;;
        --camera)
            CAMERA_SET=true
            if [[ $((i+1)) -lt ${#ARGS[@]} ]]; then
                if [[ "${ARGS[i+1]}" == "static" ]]; then
                    STATIC_RENDER=true
                fi
            fi
            ;;
        --debug|-d)
            DEBUG_ENABLED=true
            ;;
        --gpu|-g)
            GPU_ENABLED=true
            ;;
    esac
done

# If --camera is not specified at all, assume static render.
if ! $CAMERA_SET; then
    STATIC_RENDER=true
fi

# Run local binary.
if [[ "$MODE" == "local" ]]; then
    if [[ ! -x "$LOCAL_BIN" ]]; then
        echo "[ERROR] Local binary '$LOCAL_BIN' not found or not executable."
        exit 1
    fi
    echo "[INFO] Running raytracer locally..."
    "$LOCAL_BIN" "${ARGS[@]}"
    exit $?
fi

# Run cloud binary.
if [[ "$MODE" == "cloud" ]]; then
    echo "[INFO] Running raytracer on cloud instance..."

    ssh cloudgpu "[[ -x $CLOUD_BIN ]]" || {
        echo "[ERROR] Cloud binary '$CLOUD_BIN' not found or not executable."
        exit 1
    }

    # Run remotely and stream output, ensuring correct working directory.
    ssh cloudgpu "cd /workspace/real-time-ray-tracing-engine && $CLOUD_BIN ${ARGS[@]} 2>&1"
    REMOTE_EXIT_CODE=$?

    if $DEBUG_ENABLED; then
        echo "[INFO] Copying debug logs from cloud to local..."
        mkdir -p logs

        if $GPU_ENABLED; then
            DEBUG_FILES=("cuda_world_debug.json" "cuda_lights_debug.json" "cuda_scene_complexity_debug.txt" "cuda_context_debug.json")
        else
            DEBUG_FILES=("cpu_world_debug.json" "cpu_lights_debug.json")
        fi

        for FILE in "${DEBUG_FILES[@]}"; do
            REMOTE_PATH="/workspace/real-time-ray-tracing-engine/logs/$FILE"
            LOCAL_PATH="logs/$FILE"
            scp -q cloudgpu:"$REMOTE_PATH" "$LOCAL_PATH" >/dev/null 2>&1 && \
                echo "[INFO] Copied $FILE" || \
                echo "[WARN] $FILE not found on cloud."
        done
    fi

    if [[ $REMOTE_EXIT_CODE -ne 0 ]]; then
        echo "[ERROR] Cloud raytracer exited with status $REMOTE_EXIT_CODE."
        exit $REMOTE_EXIT_CODE
    fi

    if $STATIC_RENDER; then
        echo "[INFO] Copying output/$OUTPUT_FILE from cloud to local..."
        mkdir -p output
        scp -q cloudgpu:/workspace/real-time-ray-tracing-engine/output/"$OUTPUT_FILE" output/"$OUTPUT_FILE" >/dev/null 2>&1 || {
            echo "[WARN] Failed to copy output file from cloud."
        }
        echo "[INFO] Output image saved locally to output/$OUTPUT_FILE."
    else
        echo "[INFO] Skipping output copy (camera not static)."
    fi

    exit 0
fi

echo "[ERROR] Unknown mode: $MODE."
exit 1
