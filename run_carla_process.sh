# Default values
DATA_BASE_PATH='./data/Cartesian'
SAVE_BASE_PATH='./data'
CONFIG_PATH='./Tools/data_process/carla.yaml'
METHOD='max'

# Function to display usage
usage() {
    echo "Usage: $0 -q quantize_size [-d data_base_path] [-s save_base_path] [-c config_path] [-m method]"
    echo "  -q quantize_size  : Set the quantize size (required, format: x,y,z)"
    echo "  -d data_base_path : Set the base path for data (optional, default: './data/Cartesian')"
    echo "  -s save_base_path : Set the base path for saving output (optional, default: './data')"
    echo "  -c config_path    : Set the config file path (optional, default: './Tools/data_process/carla.yaml')"
    echo "  -m method         : Set the resampling method (optional, default: 'max')"
    exit 1
}

# Parse the command-line arguments
while getopts "q:d:s:c:m:" opt; do
    case ${opt} in
        q)
            IFS=',' read -r -a QUANTIZE_SIZE <<< "${OPTARG}"
            ;;
        d)
            DATA_BASE_PATH=${OPTARG}
            ;;
        s)
            SAVE_BASE_PATH=${OPTARG}
            ;;
        c)
            CONFIG_PATH=${OPTARG}
            ;;
        m)
            METHOD=${OPTARG}
            ;;
        *)
            usage
            ;;
    esac
done

# Check if quantize_size was provided
if [ -z "${QUANTIZE_SIZE}" ]; then
    echo "Error: quantize_size is required."
    usage
fi

# Run the Python script with the provided arguments
python3 Tools/data_process/carla_process.py \
    --quantize_size "${QUANTIZE_SIZE[@]}" \
    --data_base_path "${DATA_BASE_PATH}" \
    --save_base_path "${SAVE_BASE_PATH}" \
    --config_path "${CONFIG_PATH}" \
    --method "${METHOD}"
