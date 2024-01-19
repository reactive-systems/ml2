# Utility script to build Docker images

REGISTRY="${ML2_CONTAINER_REGISTRY:-ml2}"

help() { echo "Usage: $0 [-h --help] [-m --multi-arch] [--platform <string>] [--push] TOOL [DOCKERFILE_PREFIX]"; }

BUILDER="docker build"
PLATFORM=""
FILENAME="Dockerfile"
OUTPUT=""

POSITIONAL_ARGS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            help
            exit 0
            ;;
        -m|--multi-arch)
            BUILDER="docker buildx build"
            if [[ $PLATFORM == "" ]] ; then PLATFORM="--platform linux/amd64,linux/arm64" ; fi
            ;;
        --platform)
            PLATFORM="--platform $2"
            shift
            ;;
        --push)
            OUTPUT="--output type=registry"
            ;;
        -*|--*)
            echo "Unknown option $1"
            help
            exit 1
            ;;
        *)
            POSITIONAL_ARGS+=("$1")
            ;;
    esac
    shift
done

set -- "${POSITIONAL_ARGS[@]}"

TAG=$REGISTRY

if [[ $1 != "ml2" ]]; then
    TAG=$REGISTRY/$1
fi

if [[ -n $2 ]]; then
    FILENAME="$2.$FILENAME"
    if [[ $1 == "ml2" ]] || [[ $1 == "deps" ]] || [[ $1 == "neurosynt-grpc-server" ]]; then    
        TAG=$TAG:$2
    else
        TAG=$TAG-$2:latest
    fi
else
    TAG=$TAG:latest
fi

BUILD_CMD="$BUILDER --build-arg CONTAINER_REGISTRY=$REGISTRY $PLATFORM $OUTPUT -f ../docker/$1/$FILENAME  -t $TAG .."

echo "Running build command:\n\n$BUILD_CMD\n"

$BUILD_CMD
